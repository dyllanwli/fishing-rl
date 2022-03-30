import base64
import imageio
import IPython
import PIL.Image


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def create_policy_eval_video(eval_env, policy, mode, filename="eval_video", num_episodes=1000, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            obs = eval_env.reset()
            action, _states = policy.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            video.append_data(eval_env.render(mode=mode))
            if done:
                obs = eval_env.reset()
                video.append_data(eval_env.render(mode=mode))
        eval_env.close()
    return embed_mp4(filename)