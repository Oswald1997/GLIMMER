# GLIMMER


1. Download [`en-70k-0.2.lm`](https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/US%20English/) and put it in `resources` folder.

2. Run the following instruction to get `word2vec.txt` and put it in `resources` folder.
```
cat word2vec.tar.gz* | tar -xzv
```

3. Please run `main.py` under Python3.7 to use GLIMMER for unsupervised multi-document summarization.
