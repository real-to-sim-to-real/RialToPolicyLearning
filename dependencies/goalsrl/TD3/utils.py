import numpy as np

def image_float2int(image):
  return (image * 255).astype(np.uint8)

def image_int2float(image):
  return image.astype(float) / 255.0

def is_image(x):
  return len(x.shape) == 3

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6, max_lookahead=None, prob_self_goal=0.3, prob_random_goal=0.1):
        self._max_lookahead = max_lookahead
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self._prob_self_goal = prob_self_goal
        self._prob_random_goal = prob_random_goal

    def add(self, data):
        observation, observed_goal, next_observation, action, goal, reward, done = data
        if is_image(observation):
          observation = image_float2int(observation)
          observed_goal = image_float2int(observed_goal)
          next_observation = image_float2int(next_observation)
          goal = image_float2int(goal)

        data = observation, observed_goal, next_observation, action, goal, reward, done
        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        index_list = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, g, r, d = [], [], [], [], [], []

        for index in index_list:
            X1, G, Y1, U, _, _, D = self.storage[index]
            if np.random.random() < self._prob_self_goal:
              lookahead = 0
              R = 1.0
              X2 = G
              D = 0.0 # 1.0
            else:
              if np.random.rand() < self._prob_random_goal:
                _, _, _, _, X2, _, _ = self.storage[index]
              else:
                lookahead = np.random.randint(1, self._max_lookahead)
                goal_index = int((index + lookahead) % len(self.storage))
                _, X2, _, _, _, _, _ = self.storage[goal_index]
              R = 0
              D = 0.0

            x.append(np.array(X1, copy=False))
            y.append(np.array(Y1, copy=False))
            g.append(np.array(X2, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        x, y, u, g, r, d = np.array(x), np.array(y), np.array(u), np.array(g), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
        if is_image(x[0]):
          x = image_int2float(x)
          y = image_int2float(y)
          g = image_int2float(g)
        return x, y, u, g, r, d


class ExpertReplayBuffer(ReplayBuffer):
  def sample(self, batch_size):
    index_list = np.random.randint(0, len(self.storage), size=batch_size)
    x, g, u = [], [], [],

    for index in index_list:
      X1, _, _, U, G, _, _ = self.storage[index]
      x.append(np.array(X1, copy=False))
      g.append(np.array(G, copy=False))
      u.append(np.array(U, copy=False))
      
    x, g, u = np.array(x), np.array(g), np.array(u)
    if is_image(x[0]):
      x = image_int2float(x)
      g = image_int2float(g)
    return x, g, u
