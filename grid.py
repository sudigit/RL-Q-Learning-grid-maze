import numpy as np
import cv2 

columns_no=6
rows_no=6



#initializing rewards
rewards=np.full((rows_no, columns_no), -1)

walls = [(1, 1), (2, 3), (3, 4), (4, 1)]
for (x, y) in walls:
    rewards[x][y] = -100    ##wall

rewards[4][3] = 100     #target box


#define actions
actions = {
    0: "up",
    1: "down",
    2: "left",
    3: "right"
}

#check if state is terminating (walls)
def is_terminate(x,y):
  if rewards[x][y]!=-1:
    return True
  else:
    return False


#get random non-terminal starting point
def get_starting():
  #return(0,0)
  #return (3,3)
  #return (4,0)
  x = np.random.randint(0, rows_no)
  y = np.random.randint(0, columns_no)

  while(rewards[x][y]!=-1):
    x = np.random.randint(0, rows_no)
    y = np.random.randint(0, columns_no)
  return (x,y)



#initialize q values to 0
q_table = np.zeros((rows_no, columns_no, 4))


#hyperparameters
epsilon=0.9  #greedy
gamma= 0.5 #discount factor
alpha= 0.1 #learning rate


#choose action for current state
def get_action(x,y):
  if np.random.rand() < epsilon:
      return np.random.randint(0, 4)  #explore
  else:
      return np.argmax(q_table[x, y])   #exploit

#change position (if going out of grid, the position remains same)
def get_newstates(x,y,a):
  if a==0:  #up
    if x==0:
      return (x,y)
    return (x-1,y)
  elif a==1:  #down
    if x==rows_no-1:
      return (x,y)
    return (x+1,y)
  elif a==2:  #left
    if y==0:
      return (x,y)
    return (x,y-1)
  elif a==3:  #right
    if y==columns_no-1:
      return (x,y)
    return (x,y+1)


#start an episode
def start(x,y):
  path=[]
  path.append((x,y))

  while not is_terminate(x,y):
    a=get_action(x,y)
    b=get_newstates(x,y,a)
    q_table[x][y][a]+= alpha * (rewards[b[0]][b[1]] + gamma * np.max(q_table[b[0], b[1]]) - q_table[x, y, a])
    path.append((b[0],b[1]))
    x=b[0]
    y=b[1]

  return path

for episode in range(0,10000):
    x, y = get_starting()
    path = start(x, y)

    epsilon = max(0.1, epsilon * 0.995)

    #if episode % 100 == 0:
    #print(f"Episode {episode}, last path length: {len(path)}")
    #if rewards[path[-1][0]][path[-1][1]]==-100:
      #print(path)
      #print("wall")

path1=start(0,0)   #sample path for starting from point (0,0)
while rewards[path1[-1][0]][path[-1][1]]==-100:
    path1=start(0,0)
print(path1)

grid_image = np.ones((80*rows_no, 80*columns_no, 3), dtype=np.uint8) * 255
for (i1, j1) in walls:
  for i in range(80*i1,80*(i1+1)):
      for j in range(80*j1, 80*(j1+1)):
        grid_image[i,j]=(0,0,0)

for i in range(80*4,80*5):
      for j in range(80*3, 80*4):
        grid_image[i,j]=(0,255,0)

#cv2.imshow("Grid ", grid_image)

for i in range(len(path1) - 1):
    p1 = path1[i]
    p2 = path1[i + 1]

    # Convert (row, col) to (x, y) pixel coordinates
    pt1 = (p1[1] * 80 + 40, p1[0] * 80 + 40)  # center of cell
    pt2 = (p2[1] * 80 + 40, p2[0] * 80 + 40)

    cv2.line(grid_image, pt1, pt2, color=(0, 0, 255), thickness=5)

cv2.imshow("Path", grid_image)


cv2.waitKey(0)
cv2.destroyAllWindows()