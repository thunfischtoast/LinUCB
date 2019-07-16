# LinUCB
Contextual bandit algorithm called LinUCB / Linear Upper Confidence Bounds as proposed by Li, Langford and Schapire.

We implemented the two version, one with disjoint and and one with hybrid linear models, as mentioned in the paper.

See [src/de/thunfischtoast/BanditTest.java](https://github.com/thunfischtoast/LinUCB/blob/master/src/de/thunfischtoast/BanditTest.java) for basic usage example as inspired by http://john-maxwell.com/post/2017-03-17/ .


Reference:
```
@inproceedings{li2010contextual,
  title={A contextual-bandit approach to personalized news article recommendation},
  author={Li, Lihong and Chu, Wei and Langford, John and Schapire, Robert E},
  booktitle={Proceedings of the 19th international conference on World wide web},
  pages={661--670},
  year={2010},
  organization={ACM}
}
```
