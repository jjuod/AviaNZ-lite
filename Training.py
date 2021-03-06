
# This is part of the AviaNZ interface
# Holds most of the code for training CNNs

# Version 3.0 14/09/20
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2020

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# The separated code for CNN training

import os

import numpy as np

import SupportClasses
import Segment, WaveletSegment
import AviaNZ_batch


class CNNtest:

    def __init__(self,testDir,currfilt,filtname,configdir,filterdir,CLI=False):
        """ currfilt: the recognizer to be used (dict) """
        self.testDir = testDir
        self.outfile = open(os.path.join(self.testDir, "test-results.txt"),"w")

        self.currfilt = currfilt
        self.filtname = filtname

        self.configdir = configdir
        self.filterdir = filterdir
        # Note: this is just the species name, unlike the self.species in Batch mode
        species = self.currfilt['species']
        self.sampleRate = self.currfilt['SampleRate']
        self.calltypes = []
        for fi in self.currfilt['Filters']:
            self.calltypes.append(fi['calltype'])

        self.outfile.write("Recogniser name: %s\n" %(filtname))
        self.outfile.write("Species name: %s\n" % (species))
        self.outfile.write("Using data: %s\n" % (self.testDir))

        # 0. Generate GT files from annotations in test folder
        self.manSegNum = 0
        self.window = 1
        inc = None
        print('Generating GT...')
        for root, dirs, files in os.walk(self.testDir):
            for file in files:
                wavFile = os.path.join(root, file)
                if file.lower().endswith('.wav') and os.stat(wavFile).st_size != 0 and file + '.data' in files:
                    segments = Segment.SegmentList()
                    segments.parseJSON(wavFile + '.data')
                    self.manSegNum += len(segments.getSpecies(species))
                    # Currently, we ignore call types here and just
                    # look for all calls for the target species.
                    segments.exportGT(wavFile, species, window=self.window, inc=inc)

        if self.manSegNum == 0:
            print("ERROR: no segments for species %s found" % species)
            self.text = 0
            return

        # 1. Run Batch Processing upto WF and generate .tempdata files (no post-proc)
        avianz_batch = AviaNZ_batch.AviaNZ_batchProcess(parent=None, configdir=self.configdir, mode="test",
                                                        sdir=self.testDir, recogniser=filtname, wind=True)

        # 2. Report statistics of WF followed by general post-proc steps (no CNN but wind-merge neighbours-delete short)
        self.text = self.getSummary(avianz_batch, CNN=False)

        # 3. Report statistics of WF followed by post-proc steps (wind-CNN-merge neighbours-delete short)
        if "CNN" in self.currfilt:
            cl = SupportClasses.ConfigLoader()
            filterlist = cl.filters(self.filterdir, bats=False)
            CNNDicts = cl.CNNmodels(filterlist, self.filterdir, [filtname])
            if filtname in CNNDicts.keys():
                CNNmodel = CNNDicts[filtname]
                self.text = self.getSummary(avianz_batch, CNN=True, CNNmodel=CNNmodel)
            else:
                print("ERROR: Couldn't find a matching CNN!")
                self.outfile.write("No matching CNN found!\n")
                self.outfile.write("-- End of testing --\n")
                self.outfile.close()
                return
        self.outfile.write("-- End of testing --\n")
        self.outfile.close()

        print("Testing output written to " + os.path.join(self.testDir, "test-results.txt"))

        # Tidy up
        for root, dirs, files in os.walk(self.testDir):
            for file in files:
                if file.endswith('.tmpdata'):
                    os.remove(os.path.join(root, file))

    def getOutput(self):
        return self.text

    def findCTsegments(self, file, calltypei):
        calltypeSegments = []
        species = self.currfilt["species"]
        if file.lower().endswith('.wav') and os.path.isfile(file + '.tmpdata'):
            segments = Segment.SegmentList()
            segments.parseJSON(file + '.tmpdata')
            if len(self.calltypes) == 1:
                ctSegments = segments.getSpecies(species)
            else:
                ctSegments = segments.getCalltype(species, self.calltypes[calltypei])
            for indx in ctSegments:
                seg = segments[indx]
                calltypeSegments.append(seg[:2])

        return calltypeSegments

    def getSummary(self, avianz_batch, CNN=False, CNNmodel=None):
        autoSegNum = 0
        autoSegCT = [[] for i in range(len(self.calltypes))]
        ws = WaveletSegment.WaveletSegment()
        TP = FP = TN = FN = 0
        for root, dirs, files in os.walk(self.testDir):
            for file in files:
                wavFile = os.path.join(root, file)
                if file.lower().endswith('.wav') and os.stat(wavFile).st_size != 0 and \
                        file + '.tmpdata' in files and file[:-4] + '-res' + str(float(self.window)) + 'sec.txt' in files:
                    autoSegCTCurrent = [[] for i in range(len(self.calltypes))]
                    avianz_batch.filename = os.path.join(root, file)
                    avianz_batch.loadFile([self.filtname], anysound=False)
                    duration = int(np.ceil(len(avianz_batch.audiodata) / avianz_batch.sampleRate))
                    for i in range(len(self.calltypes)):
                        ctsegments = self.findCTsegments(avianz_batch.filename, i)
                        post = Segment.PostProcess(configdir=self.configdir, audioData=avianz_batch.audiodata,
                                                   sampleRate=avianz_batch.sampleRate,
                                                   tgtsampleRate=self.sampleRate, segments=ctsegments,
                                                   subfilter=self.currfilt['Filters'][i], CNNmodel=CNNmodel, cert=50)
                        post.wind()
                        if CNN and CNNmodel:
                            post.CNN()
                        if 'F0' in self.currfilt['Filters'][i] and 'F0Range' in self.currfilt['Filters'][i]:
                            if self.currfilt['Filters'][i]["F0"]:
                                print("Checking for fundamental frequency...")
                                post.fundamentalFrq()
                        post.joinGaps(maxgap=self.currfilt['Filters'][i]['TimeRange'][3])
                        post.deleteShort(minlength=self.currfilt['Filters'][i]['TimeRange'][0])
                        if post.segments:
                            for seg in post.segments:
                                autoSegCTCurrent[i].append(seg[0])
                                autoSegCT[i].append(seg[0])
                                autoSegNum += 1
                    # back-convert to 0/1:
                    det01 = np.zeros(duration)
                    for i in range(len(self.calltypes)):
                        for seg in autoSegCTCurrent[i]:
                            det01[int(seg[0]):int(seg[1])] = 1
                    # get and parse the agreement metrics
                    GT = self.loadGT(os.path.join(root, file[:-4] + '-res' + str(float(self.window)) + 'sec.txt'),
                                     duration)
                    _, _, tp, fp, tn, fn = ws.fBetaScore(GT, det01)
                    TP += tp
                    FP += fp
                    TN += tn
                    FN += fn
        # Summary
        total = TP + FP + TN + FN
        if total == 0:
            print("ERROR: failed to find any testing data")
            return

        if TP + FN != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        if TP + FP != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if TN + FP != 0:
            specificity = TN / (TN + FP)
        else:
            specificity = 0
        accuracy = (TP + TN) / (TP + FP + TN + FN)

        if CNN:
            self.outfile.write("\n\n-- Wavelet Pre-Processor + CNN detection summary --\n")
        else:
            self.outfile.write("\n-- Wavelet Pre-Processor detection summary --\n")
        self.outfile.write("TP | FP | TN | FN seconds:\t %.2f | %.2f | %.2f | %.2f\n" % (TP, FP, TN, FN))
        self.outfile.write("Specificity:\t\t%.2f %%\n" % (specificity * 100))
        self.outfile.write("Recall (sensitivity):\t%.2f %%\n" % (recall * 100))
        self.outfile.write("Precision (PPV):\t%.2f %%\n" % (precision * 100))
        self.outfile.write("Accuracy:\t\t%.2f %%\n\n" % (accuracy * 100))
        self.outfile.write("Manually labelled segments:\t%d\n" % (self.manSegNum))
        for i in range(len(self.calltypes)):
            self.outfile.write("Auto suggested \'%s\' segments:\t%d\n" % (self.calltypes[i], len(autoSegCT[i])))
        self.outfile.write("Total auto suggested segments:\t%d\n\n" % (autoSegNum))

        if CNN:
            text = "Wavelet Pre-Processor + CNN detection summary\n\n\tTrue Positives:\t%d seconds (%.2f %%)\n\tFalse Positives:\t%d seconds (%.2f %%)\n\tTrue Negatives:\t%d seconds (%.2f %%)\n\tFalse Negatives:\t%d seconds (%.2f %%)\n\n\tSpecificity:\t%.2f %%\n\tRecall:\t\t%.2f %%\n\tPrecision:\t%.2f %%\n\tAccuracy:\t%.2f %%\n" \
                   % (TP, TP * 100 / total, FP, FP * 100 / total, TN, TN * 100 / total, FN, FN * 100 / total,
                      specificity * 100, recall * 100, precision * 100, accuracy * 100)
        else:
            text = "Wavelet Pre-Processor detection summary\n\n\tTrue Positives:\t%d seconds (%.2f %%)\n\tFalse Positives:\t%d seconds (%.2f %%)\n\tTrue Negatives:\t%d seconds (%.2f %%)\n\tFalse Negatives:\t%d seconds (%.2f %%)\n\n\tSpecificity:\t%.2f %%\n\tRecall:\t\t%.2f %%\n\tPrecision:\t%.2f %%\n\tAccuracy:\t%.2f %%\n" \
                   % (TP, TP * 100 / total, FP, FP * 100 / total, TN, TN * 100 / total, FN, FN * 100 / total,
                      specificity * 100, recall * 100, precision * 100, accuracy * 100)
        return text

    def loadGT(self, filename, length):
        import csv
        annotation = []
        # Get the segmentation from the txt file
        with open(filename) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        if d[-1] == []:
            d = d[:-1]
        if len(d) != length:
            print("ERROR: annotation length %d does not match file duration %d!" % (len(d), n))
            self.annotation = []
            return False

        # for each second, store 0/1 presence:
        for row in d:
            annotation.append(int(row[1]))

        return annotation

