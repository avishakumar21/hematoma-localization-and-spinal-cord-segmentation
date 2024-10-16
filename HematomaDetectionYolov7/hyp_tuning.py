from nni.experiment import Experiment

if __name__=='__main__':
    search_space = {
        'batch': {'_type': 'choice', '_value': [8, 16, 32]},
        'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
        'epochs': {'_type': 'choice', '_value': [50, 75, 100]},
    }

    exp = Experiment('local')
    exp.config.trial_concurrency = 1
    exp.config.max_trial_number = 50
    exp.config.search_space = search_space
    exp.config.trial_command = 'python train.py'
    exp.config.trial_code_directory = '.'
    exp.config.tuner.name = 'TPE'
    exp.config.tuner.class_args = {'optimize_mode': 'maximize'}

    exp.run(8085)

    input("Press Enter to continue...")
    exp.stop()
