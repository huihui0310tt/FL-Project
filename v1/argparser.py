def get_arg():
    ##############################################################     # 初步設定
    user = 7  # Client

    round = 40  # Round
    epochs = 5  # Epr  (Epoch per Round)
    lr = 0.01
    batch_size = 64

    global_model = None
    # 如果需要其他外部的preTrained Model, 請在此處load進global_model

    no_cuda = False  # False = use cuda,       True = use cpu
    # no_cuda ( True )  = 使用cpu做training
    # no_cuda ( False ) = 使用cuda做training

    mode = 2
    # mode 1 = 開thread做Client模擬 (平行) (if gpu memory足夠)
    # mode 2 = 使用iterator做Client模擬 (輪流)

    return user, round, epochs, lr, batch_size, global_model, no_cuda, mode
##########################################################################
# 請在/data根據規格擺置足夠數量client的data  (編號從edge1開始)
# example : edge1
#     /Harmonia
#     ├─ ..
#     ├─ aggregator_base.py
#     ├─ client_base.py
#     ├─ argparser.py
#     ├─ main.py
#     ├─ net.py
#     ├─ model_save
#           ├─  ( None )
#     ├─ data
#           ├─ edge1
#                 ├─ 1 Surprise
#                       ├─ 1.jpg
#                       ├─ 2.jpg
#                       ├─ ..
#                 ├─ 2 Fear
#                       ├─ 1.jpg
#                       ├─ 2.jpg
#                       ├─ ..
#                 ├─ 3 Disgust
#                       ├─ 1.jpg
#                       ├─ 2.jpg
#                       ├─ ..
#                 ├─ 4 Happiness
#                       ├─ 1.jpg
#                       ├─ 2.jpg
#                       ├─ ..
#                 ├─ 5 Sadness
#                       ├─ 1.jpg
#                       ├─ 2.jpg
#                       ├─ ..
#                 ├─ 6 Anger
#                       ├─ 1.jpg
#                       ├─ 2.jpg
#                       ├─ ..
#                 ├─ 7 Neutral
#                       ├─ 1.jpg
#                       ├─ 2.jpg
#                       ├─ ..
#                 ├─ test
#                       ├─ 1.jpg
#                       ├─ 2.jpg
#                       ├─ ..
#
