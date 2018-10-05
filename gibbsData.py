import codecs
import os

from preprocessing import wikiUtils, UtilNews
#from preprocessing import wikiUtils, UtilNews, init_scws


class WikiStream(object):

    def __init__(self, input_file):
        self._inputFile = input_file

    def __iter__(self):
        with open(self._inputFile, 'r') as fi:
            for doc in fi:
                tokens = wikiUtils().process_wiki(doc)
                if len(tokens) < 10:
                    continue
                yield tokens


def wiki_length(stream_data):
    length = sum(1 for _ in stream_data)
    return str(length)


def save_wiki(num_docs, stream_corpus, fname):
    dest_file = codecs.open(fname, 'w', encoding='utf-8', errors='ignore')
    dest_file.write(num_docs + '\n')
    #print>> dest_file, num_docs
    for doc in stream_corpus:
        for w in doc:
            dest_file.write(w+' ')
        dest_file.write('\n')
            #print>> dest_file, w,
        #print>> dest_file
    dest_file.close()



def parse_News(rootdir):
    labels = {}
    class_num_docs = {}
    index = 1
    for dirpath, dirname, filename in os.walk(rootdir):
        num_docs = 0
        for file in filename:
            label_name = os.path.basename(os.path.normpath(dirpath))
            file_path = os.path.join(dirpath, file)

            if label_name not in labels:
                labels[label_name] = index
                index +=1

            if label_name not in class_num_docs:
                class_num_docs[label_name] = 0

            file_tokens = UtilNews().process_msg(file_path)
            if len(file_tokens)<10:
                continue

            num_docs +=1
            class_num_docs[label_name] = num_docs
    return labels, class_num_docs, sum(class_num_docs.values())



def saveNews(labels, dirName, num_docs, dataf, labelf):
    data_file = codecs.open(dataf,'w', encoding='utf-8',errors='ignore')
    label_file = open(labelf, 'w')

    total_numdocs = str(num_docs)
    print>>data_file,total_numdocs
    for dirpath, dirname, filename in os.walk(dirName):

        for file in filename:
            label_name = os.path.basename(os.path.normpath(dirpath))
            file_path = os.path.join(dirpath, file)

            label_id = labels[label_name]
            file_tokens = UtilNews().process_msg(file_path)
            if len(file_tokens)<10:
                continue

            for token in file_tokens:
                print>>data_file,token,
            print>>data_file

            print>>label_file, label_id

    data_file.close()
    label_file.close()



def create_wiki_train(ifile, ofile):
    stream = WikiStream(ifile) #'/home/nooshin/Downloads/Socher_wiki/WestburyLab.wikicorp.201004.txt'
    numDocs = wiki_length(stream)
    save_wiki(numDocs, stream, ofile) #/tmp/wikiGibbs.dat

def create_wiki_test(ofile):
    dlist, numDocs, _, _, _ = init_scws()
    save_wiki(numDocs, dlist, ofile) #'/tmp/scwsGibbs.dat'

def create_News_train(rootdir, data_ofile, label_ofile):
    #rootdir = '/home/nooshin/MVWE_Datasets/20News/20news-bydate-train/'
    labels, c, n = parse_News(rootdir)
    saveNews(labels, rootdir, n, data_ofile, label_ofile) #'/tmp/news_train.dat', /tmp/train.label

def create_News_test(rootdir, data_ofile, label_ofile):
    #rootdir = '/home/nooshin/MVWE_Datasets/20News/20news-bydate-test/'
    labels, c, n = parse_News(rootdir)
    saveNews(labels, rootdir, n, data_ofile, label_ofile) #'/tmp/news_test.dat' , '/tmp/test.label'


def create_wikiGibbsData(train_ifile, trainGibs, testGibs):
    create_wiki_train(train_ifile, trainGibs)
    create_wiki_test(testGibs)


def create_NewsGibbsData(tr_dir, tr_data, tr_label, te_dir, te_data, te_label):
    create_News_train(tr_dir, tr_data, tr_label)
    create_News_test(te_dir, te_data, te_label)



#create_wiki_train('/home/nooshin/Downloads/Socher_wiki/WestburyLab.wikicorp.201004.txt','/tmp/wikiGibbs.dat')

#print ('done')
#create_wikiGibbsData('/home/nooshin/Downloads/Socher_wiki/WestburyLab.wikicorp.201004.txt', '/tmp/wikiGibbs.dat','/tmp/scwsGibbs.dat')

#create_NewsGibbsData('/home/nooshin/MVWE_Datasets/20News/20news-bydate-train/','/tmp/news_train.dat','/tmp/train.label',
#                         '/home/nooshin/MVWE_Datasets/20News/20news-bydate-test/','/tmp/news_test.dat' , '/tmp/test.label')


# s = [['nooshin','mojab'],['hello','happy','positive']]
# des = open('/tmp/test.txt', 'w')
# des.write('2'+ '\n')
# for doc in s:
#     for w in doc:
#         des.write(w+' ')
#     des.write('\n')
# des.close()
