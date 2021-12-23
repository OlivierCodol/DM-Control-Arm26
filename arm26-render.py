from acme import wrappers
import numpy as np
import imageio
import arm26


def render(env):
    return env.physics.render(camera_id=0)


def display_video(frames, filename='tmp.mp4'):
    """Save video."""
    with imageio.get_writer(filename, fps=60) as video:
        for frame in frames:
            video.append_data(frame)


environment = arm26.load('easy', task_kwargs={'time_limit': 2}, visualize_reward=True)
environment = wrappers.SinglePrecisionWrapper(environment)

muscle_names = ["SF", "SE", "EF", "EE", "BF", "BE"]

for k in range(6):
    timestep = environment.reset()
    frame_stack = [render(environment)]

    # stimulate one muscle only
    a = [0, 0, 0, 0, 0, 0]
    a[k] = 1

    # render resuting simulation
    while not timestep.last():
        action = np.ones((1, 6)) * a
        timestep = environment.step(action)
        frame_stack.append(render(environment))
    display_video(np.array(frame_stack)[::2, :, :, :], filename='./arm26-video-runs/' + muscle_names[k] + '.gif')
