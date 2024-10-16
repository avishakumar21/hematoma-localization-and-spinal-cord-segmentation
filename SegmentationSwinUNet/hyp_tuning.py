from nni.experiment import Experiment

if __name__=='__main__':
    search_space = {
        'batch': {'_type': 'choice', '_value': [16, 24, 32]},
        'lr': {'_type': 'loguniform', '_value': [0.001, 0.1]},
        'epochs': {'_type': 'choice', '_value': [100, 150, 200]} #20, 40, 60 
    }

    exp = Experiment('local')
    exp.config.trial_concurrency = 1
    exp.config.max_trial_number = 50
    exp.config.search_space = search_space
    exp.config.trial_command = 'python train.py'
    exp.config.trial_code_directory = '.'
    exp.config.tuner.name = 'TPE'
    exp.config.tuner.class_args = {'optimize_mode': 'maximize'}

    exp.run(8082)

    input("Press Enter to continue...")
    exp.stop()

    ####### Localhost address #######
    # http://127.0.0.1:8082/detail
    #################################