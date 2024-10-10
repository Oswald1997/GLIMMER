# GLIMMER

This repository contains code for the paper 'GLIMMER: Incorporating Graph and Lexical Features in Unsupervised Multi-Document Summarization', which has been accepted by ECAI 2024.

## Code

1. Download [`en-70k-0.2.lm`](https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/US%20English/) and put it in `resources` folder.

2. Run the following instruction to get `word2vec.txt` and put it in `resources` folder.
```
cat word2vec.tar.gz* | tar -xzv word2vec.txt
```

3. Please run `main.py` under Python3.7 to use GLIMMER for unsupervised multi-document summarization.

## Citation

Please cite if you find this paper or repo useful:
```bibtex
@misc{liu2024glimmerincorporatinggraphlexical,
      title={GLIMMER: Incorporating Graph and Lexical Features in Unsupervised Multi-Document Summarization}, 
      author={Ran Liu and Ming Liu and Min Yu and Jianguo Jiang and Gang Li and Dan Zhang and Jingyuan Li and Xiang Meng and Weiqing Huang},
      year={2024},
      url={https://arxiv.org/abs/2408.10115}, 
}
```

