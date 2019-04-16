import os
import sys
import multiprocessing
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    """ 
        os.path.basename(arg): 去掉路径名，返回文件名
        arg是一个文件的路径名
        
        关于sys.argv()的解释：http://www.cnblogs.com/aland-1415/p/6613449.html
        sys.argv[0]：返回代码文件的本身的路径
    """
    program = os.path.basename(sys.argv[0])

    """
        logging是python内置的用于打印日志的类
        参考：https://www.cnblogs.com/CJOKER/p/8295272.html
    """
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 4:
        print("Using: python train_w2v.py one-billion-word-benchmark output_gensim_model output_word_vector")
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]

    """
        关于gensim中Word2Vec函数的讲解：https://www.cnblogs.com/pinard/p/7278324.html
        这里使用了word2vec提供的LineSentence类来读文件，其中workers参数是指线程数
    """
    model = Word2Vec(LineSentence(inp), size=150, window=6, min_count=2, workers=(multiprocessing.cpu_count()-2), hs=1, sg=1, negative=10)

    model.save(outp1)      # 保存训练好的模型
    model.wv.save_word2vec_format(outp2, binary=True)    # 将训练好的word embedding向量存入outp2文件中
    # 从outp2文件中读取embedding向量时用gensim的KeyedVectors


