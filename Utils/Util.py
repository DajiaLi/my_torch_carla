import numpy as np
import cv2
import matplotlib.pyplot as plt

def abstract_data_numpy(x, interval_num):
    '''
    输入像素在0-1
    input x.shape: # (h, w, 3)
    output result.shape :(2 * h, w, 3)
    '''
    step = (1 - (-1)) / interval_num
    k = (x - (-1)) / step
    k = np.floor(k)
    x_lower = -1 + k * step
    x_lower = np.clip(x_lower, -1, 1).astype(np.float32)
    x_upper = x_lower + step
    x_upper = np.clip(x_upper, -1, 1).astype(np.float32)
    # if x=1，it should not be abstracted to [1,1]
    eq = np.equal(x_lower, x_upper)
    x_lower = x_lower - step * eq
    x_lower = np.clip(x_lower, -1, 1)

    result = np.concatenate([x_lower, x_upper], axis=0).astype(np.float32)
    return result

def draw_linegraph_between_two(txt_noabstract, txt_abstract, save_name="res.png", fig_name="fig", min_reward=None, max_reward=None):
    content_noabstract = None
    content_abstract = None
    with open(txt_noabstract, 'r', encoding='utf-8') as f1:
        content_noabstract = f1.read().split('\n')
    with open(txt_abstract, 'r', encoding='utf-8') as f2:
        content_abstract = f2.read().split('\n')

    # 过滤无关信息
    content_noabstract = [line for line in content_noabstract if line.startswith("episode")]
    content_abstract = [line for line in content_abstract if line.startswith("episode")]

    min_len = min(len(content_noabstract), len(content_abstract))

    content_noabstract = content_noabstract[:min_len]
    content_abstract = content_abstract[:min_len]

    episodes = []
    epsilon = []
    episode_reward_noabstract = []
    episode_reward_abstract = []

    for i in range(min_len):
        info1 = content_noabstract[i].split(',')
        episodes.append(int(info1[0][len("episode "):]))
        epsilon.append(float(info1[2][len("epsilon "):]))
        reward_noabstract = int(info1[1][len("episode_reward "):])
        info2 = content_abstract[i].split(',')
        reward_abstract = int(info2[1][len("episode_reward "):])
        if max_reward is not None:
            reward_noabstract = reward_noabstract if reward_noabstract < max_reward else max_reward
            reward_abstract = reward_abstract if reward_abstract < max_reward else max_reward
        if min_reward is not None:
            reward_noabstract = reward_noabstract if reward_noabstract > min_reward else min_reward
            reward_abstract = reward_abstract if reward_abstract > min_reward else min_reward
        episode_reward_noabstract.append(reward_noabstract)
        episode_reward_abstract.append(reward_abstract)

    # 第一步：创建一个figure，figsize参数设置figure的长和宽
    fig = plt.figure(figsize=(22, 5))

    # 第二步，快速创建单行或单列布局的多子图（多行多列不支持）。sharey表示共用Y轴
    ax1, ax2, ax3 = fig.subplots(1, 3, sharey=True)


    color_abstract = 'lightskyblue'
    color_noabstract = 'orange'
    color_epsilon = 'blue'
    # 第三步，逐个创建子图，一个 ax 就是一个子图
    # ax1作图，并设置X,Y轴名称，以及设置图上方的标题
    ax1.plot(episodes, episode_reward_abstract, color=color_abstract)
    ax1.set_xlabel('episode')
    ax1.set_ylabel('reward')
    ax1.set_title('abstract')
    ax1_ = ax1.twinx()  # 创建共用x轴的第二个y轴
    ax1_.set_ylabel('epsilon', color=color_epsilon)
    ax1_.plot(episodes, epsilon, color=color_epsilon)
    # ax2作图，并设置X轴名称，以及设置图上方的标题
    ax2.plot(episodes, episode_reward_noabstract, color=color_noabstract)
    ax2.plot(episodes, episode_reward_abstract, color=color_abstract)
    ax2.set_xlabel('episode')
    ax2.set_ylabel('reward')
    ax2.set_title('res')
    ax2_ = ax2.twinx()  # 创建共用x轴的第二个y轴
    ax2_.set_ylabel('epsilon', color=color_epsilon)
    ax2_.plot(episodes, epsilon, color=color_epsilon)

    # ax3作图，并设置X轴名称，以及设置图上方的标题
    ax3.plot(episodes, episode_reward_noabstract, color=color_noabstract)
    ax3.set_xlabel('episode')
    ax3.set_ylabel('reward')
    ax3.set_title('no-abstract')
    ax3_ = ax3.twinx()  # 创建共用x轴的第二个y轴
    ax3_.set_ylabel('epsilon', color=color_epsilon)
    ax3_.plot(episodes, epsilon, color=color_epsilon)

    # 第四步，设置fig的标题，比ax1，ax2，ax3的标题更高一级
    fig.suptitle(fig_name)

    # 第五步，展示图
    plt.savefig(save_name)
    plt.show()

def draw_single_linegraph(txt1, save_name=None):
    txt1_content = None
    with open(txt1, 'r', encoding='utf-8') as f1:
        txt1_content = f1.read().split('\n')

    # 过滤无关信息
    txt1_content = [line for line in txt1_content if line.startswith("episode")]

    episodes = []
    epsilon = []
    episode_reward1 = []
    episode_reward2 = []

    for i in range(len(txt1_content)):
        info1 = txt1_content[i].split(',')
        episodes.append(int(info1[0][len("episode "):]))
        epsilon.append(float(info1[2][len("epsilon "):]))
        episode_reward1.append(int(info1[1][len("episode_reward "):]))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('episode')
    ax1.set_ylabel('reward', color=color)
    ax1.plot(episodes, episode_reward1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

    color = 'tab:blue'
    ax2.set_ylabel('epsilon', color=color)
    ax2.plot(episodes, epsilon, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


if __name__ == '__main__':
    # draw_linegraph_between_two(
    #     txt_noabstract=r"../txt/dqn_train_point43_2.txt",
    #     txt_abstract=r"../txt/dqn_train_point43_abstract2.txt",
    #     fig_name="dqn_point43_2",
    #     save_name="../txt/dqn_point43_2")
    # draw_linegraph_between_two(
    #     txt_noabstract=r"../txt/dqn_train_pointdefault/dqn_train_pointdefault.txt",
    #     txt_abstract=r"../txt/dqn_train_pointdefault/dqn_train_pointdefault_abstract.txt",
    #     fig_name="dqn_pointdefault",
    #     max_reward=4000,
    #     save_name="../txt/dqn_train_pointdefault/dqn_pointdefault")
    # draw_linegraph_between_two(
    #     txt_noabstract=r"../txt/dqn_train_pointdefault2/dqn_train_pointdefault2.txt",
    #     txt_abstract=r"../txt/dqn_train_pointdefault2/dqn_train_pointdefault_abstract2.txt",
    #     fig_name="dqn_pointdefault2",
    #     min_reward=-250,
    #     max_reward=50,
    #     save_name="../txt/dqn_train_pointdefault2/dqn_pointdefault2")
    draw_linegraph_between_two(
        txt_noabstract=r"../txt/dqn_point3_2/dqn_train_point3.txt",
        txt_abstract=r"../txt/dqn_point3_2/dqn_train_point3_abstract2.txt",
        fig_name="dqn_train_point3",
        # min_reward=-250,
        # max_reward=50,
        save_name="../txt/dqn_point3_2/dqn_train_point3")
