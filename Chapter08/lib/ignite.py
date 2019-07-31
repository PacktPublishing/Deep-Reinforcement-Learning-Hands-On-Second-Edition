import ptan
import enum
import time
from typing import Optional
from ignite.engine import Engine, State
from ignite.engine import Events as EngineEvents
from ignite.handlers.timing import Timer


class EndOfEpisodeHandler:
    class Events(enum.Enum):
        EPISODE_COMPLETED = "episode_completed"
        BOUND_REWARD_REACHED = "bound_reward_reached"

    def __init__(self, exp_source: ptan.experience.ExperienceSource, alpha: float = 0.98,
                 bound_avg_reward: Optional[float] = None):
        self._exp_source = exp_source
        self._alpha = alpha
        self._bound_avg_reward = bound_avg_reward

    def attach(self, engine: Engine):
        engine.add_event_handler(EngineEvents.ITERATION_COMPLETED, self)
        engine.register_events(*self.Events)
        State.event_to_attr[self.Events.EPISODE_COMPLETED] = "episode"
        State.event_to_attr[self.Events.BOUND_REWARD_REACHED] = "episode"

    def __call__(self, engine: Engine):
        for reward, steps in self._exp_source.pop_rewards_steps():
            engine.state.episode = getattr(engine.state, "episode", 0) + 1
            engine.state.episode_reward = reward
            engine.state.episode_steps = steps
            engine.state.metrics['reward'] = reward
            engine.state.metrics['steps'] = steps
            self._update_smoothed_metrics(engine, reward, steps)
            engine.fire_event(self.Events.EPISODE_COMPLETED)
            if self._bound_avg_reward is not None and engine.state.metrics['avg_reward'] >= self._bound_avg_reward:
                engine.fire_event(self.Events.BOUND_REWARD_REACHED)

    def _update_smoothed_metrics(self, engine: Engine, reward: float, steps: int):
        for attr_name, val in zip(('avg_reward', 'avg_steps'), (reward, steps)):
            if attr_name not in engine.state.metrics:
                engine.state.metrics[attr_name] = val
            else:
                engine.state.metrics[attr_name] *= self._alpha
                engine.state.metrics[attr_name] += (1-self._alpha) * val


class EpisodeFPSHandler:
    def __init__(self, fps_mul: float = 1.0):
        self._timer = Timer(average=True)
        self._fps_mul = fps_mul
        self._started_ts = time.time()

    def attach(self, engine: Engine):
        self._timer.attach(engine, step=EngineEvents.ITERATION_COMPLETED)
        engine.add_event_handler(EndOfEpisodeHandler.Events.EPISODE_COMPLETED, self)

    def __call__(self, engine: Engine):
        t_val = self._timer.value()
        if engine.state.iteration > 1:
            engine.state.metrics['fps'] = self._fps_mul / t_val
        engine.state.metrics['time_passed'] = time.time() - self._started_ts
        self._timer.reset()


class PeriodicEvents:
    """
    The same as CustomPeriodicEvent from ignite.contrib, but use true amount of iterations,
    which is good for TensorBoard
    """
    class Events(enum.Enum):
        ITERATIONS_10_COMPLETED = "iterations_10_completed"
        ITERATIONS_100_COMPLETED = "iterations_100_completed"
        ITERATIONS_1000_COMPLETED = "iterations_1000_completed"

    INTERVAL_TO_EVENT = {
        10: Events.ITERATIONS_10_COMPLETED,
        100: Events.ITERATIONS_100_COMPLETED,
        1000: Events.ITERATIONS_1000_COMPLETED,
    }

    def attach(self, engine: Engine):
        engine.add_event_handler(EngineEvents.ITERATION_COMPLETED, self)
        engine.register_events(*self.Events)
        for e in self.Events:
            State.event_to_attr[e] = "iteration"

    def __call__(self, engine: Engine):
        for period, event in self.INTERVAL_TO_EVENT.items():
            if engine.state.iteration % period == 0:
                engine.fire_event(event)

