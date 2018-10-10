import codecs
import os

from preprocessing import wikiUtils, UtilNews



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
    for doc in stream_corpus:
        for w in doc:
            dest_file.write(w+' ')
        dest_file.write('\n')
           
        
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









