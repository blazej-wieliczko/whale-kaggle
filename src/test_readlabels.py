from unittest import TestCase
import ripdata
from spro_utils import load_mfcc_file, save_mfcc_file

__author__ = 'blazej'

class TestReadlabels(TestCase):
#    def test_readlabels(self):
#        labels = ripdata.getlabels_for_filenames(["harry", "john", "mark"], {'john' : "1", "mark" : "2", "harry" : '5'})
#
#        self.assertSequenceEqual(list(labels), [5,1,2])

    def test_load_save_spro(self):
        data = load_mfcc_file("../data/small/train/train64.fbank")
        save_mfcc_file(data, "/tmp/dupa.fbank")
        data2 = load_mfcc_file("/tmp/dupa.fbank")
        self.assertSequenceEqual(data.data, data2.data)



