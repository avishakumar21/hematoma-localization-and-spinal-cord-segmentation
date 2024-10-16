from nni.experiment import Experiment

if __name__=='__main__':
    search_space = {
        'batch': {'_type': 'choice', '_value': [4, 8, 16, 32]},
        'lr': {'_type': 'loguniform', '_value': [0.0001, 0.01]},
        'epochs': {'_type': 'choice', '_value': [20, 40, 60, 80]},
    }

    exp = Experiment('local')
    exp.config.trial_concurrency = 1
    exp.config.max_trial_number = 50
    exp.config.search_space = search_space
    exp.config.trial_command = 'python train.py'
    exp.config.trial_code_directory = '.'
    exp.config.tuner.name = 'TPE'
    exp.config.tuner.class_args = {'optimize_mode': 'maximize'}

    exp.run(8081)

    input("Press Enter to continue...")
    exp.stop()