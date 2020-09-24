import os
import glob
from pybullet_tools.utils import read_json, write_json
from pddlstream.utils import mkdir

if __name__ == '__main__':
    SKILL = 'scoop'
    MATERIAL = {'pour': 'red_men', 'scoop': 'chickpeas'}
    all_dirs = glob.glob('data/pr2_{}_*'.format(SKILL))
    all_trials = []
    for dir in all_dirs:
        trial_file = os.path.join(dir, 'trials.json')
        data = read_json(trial_file)
        for d in data:
            if d['policy'] == 'training' and d['material'] == MATERIAL[SKILL] and d['score'] is not None:
                all_trials.append(d)
    newdir = 'data/pr2_{}/'.format(SKILL)
    mkdir(newdir)
    write_json(os.path.join(newdir, 'all_trials.json'), all_trials)