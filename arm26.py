# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# ============================================================================
# Modified by O. Codol, email: codol.olivier@gmail.com
# ============================================================================

"""Arm26 domain."""

from dm_control.utils import io as resources
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite.reacher import Physics, Reacher
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
import numpy as np

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .15
_SMALL_TARGET = .045


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return resources.GetResource('arm26.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns arm26 with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Arm26(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns arm26 with sparse reward with 1e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Arm26(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


class Arm26(Reacher):
  """A arm26 `Task` to reach the target."""

  def __init__(self, target_size, random=None):
    """Initialize an instance of `Arm26`.

    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(target_size=target_size, random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    physics.named.model.geom_size['target', 0] = self._target_size
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    # Randomize target position
    angle = self.random.uniform(-2 * np.pi / 4, 1 * np.pi / 4)
    radius = self.random.uniform(0.3, 1.)
    physics.named.model.body_pos['target', 'x'] = radius * np.sin(angle)
    physics.named.model.body_pos['target', 'y'] = radius * np.cos(angle)

    super().initialize_episode(physics)


def load(task_name, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
  """Returns an Arm26 environment given a task name.

  Args:
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` specifying keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.

  Raises:
    ValueError: If the task doesn't exist.

  Returns:
    An instance of the requested environment.
  """

  if task_name not in SUITE:
    raise ValueError('Level {!r} does not exist in domain arm26.'.format(task_name))

  task_kwargs = task_kwargs or {}
  if environment_kwargs is not None:
    task_kwargs = dict(task_kwargs, environment_kwargs=environment_kwargs)
  env = SUITE[task_name](**task_kwargs)
  env.task.visualize_reward = visualize_reward
  return env