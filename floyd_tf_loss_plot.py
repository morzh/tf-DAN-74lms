

import matplotlib.pyplot as plt
import re

epn = 0
losses = list()

# fname = 'tf-loss-vals_1.txt'
# fname = 'tf-loss-ds1.txt'
fname = 'tf-loss-ds2.txt'


with open(fname) as f:
    content = f.readlines()
content = [x.strip() for x in content] 

for line in content:

    # if 'INFO:tensorflow:loss'  in line:
    if 'INFO:tensorflow:Saving dict for global step'  in line:
    # if 'Starting a training cycle'  in line:
        # epn += 1
        # m = re.search('INFO:tensorflow:loss = (.+?), step', line)
        m = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if m:
            # found = m.group(1)
            losses.append(float(m[-1]))
            # losses.append(float(found))


print losses

x_axis = range(0,len(losses))

plt.plot(x_axis, losses)
plt.show()


# print epn