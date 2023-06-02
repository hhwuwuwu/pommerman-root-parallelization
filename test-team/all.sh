# #前端位置为tensorflow中python的位置，后端位置为test-noteam中待执行文件的位置
#!/bin/sh
### Esc进入命令行运行模式
###: set ff=unix
/home/river/anaconda3/envs/tensorflow/bin/python /home/river/pommerman/pommerman-root-parallelization/test-team/2v2-mcts.py & /home/river/anaconda3/envs/tensorflow/bin/python /home/river/pommerman/pommerman-root-parallelization/test-team/2v2-pmcts_1.py & /home/river/anaconda3/envs/tensorflow/bin/python /home/river/pommerman/pommerman-root-parallelization/test-team/2v2-pmcts_2.py & /home/river/anaconda3/envs/tensorflow/bin/python /home/river/pommerman/pommerman-root-parallelization/test-team/2v2-pmcts_4.py & /home/river/anaconda3/envs/tensorflow/bin/python /home/river/pommerman/pommerman-root-parallelization/test-team/2v2-pmcts_8.py & 
