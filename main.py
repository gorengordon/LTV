from ltv_class import *
import pickle
from os import listdir
from os.path import isfile, join

process_flow = {
    'get_stats': True,
    'get_priors': True,
    'update_posteriors': False,
    'calc_ltv': False,
    'update_ranking': True,
    'check_result': True,
    'save_to_csv': True
}


past_filename = data_path + 'Ground_truth/ref-20160401.csv'
present_filename = data_path + 'Ground_truth/ref-20160701.csv'
results_path = '../../LTV/results/'

candidate_type = ['FTB', 'User']
for c_type in candidate_type:
    print('Starting ', c_type, '...')
    report_filenames = [f for f in listdir(reports_path) if isfile(join(reports_path, f)) if 'less' in f and 'Copy' in f and c_type in f]
    # report_filenames = [report_filenames[0]]

    if process_flow['get_stats']:
        print('get_stats...')
        the_class = LtvClass(filename=past_filename, target_periods=[7,30,90], max_count=100000)
        the_class.get_stats()
        with open(results_path + 'ltv_class.pkl', 'wb') as output:
            pickle.dump(the_class, output, pickle.HIGHEST_PROTOCOL)

    if process_flow['get_priors']:
        print('get_priors...')
        with open(results_path + 'ltv_class.pkl', 'rb') as input:
            the_class = pickle.load(input)
        the_class.define_priors(n_bins=100)
        the_class.update_priors()
        with open(results_path + 'ltv_class.pkl', 'wb') as output:
            pickle.dump(the_class, output, pickle.HIGHEST_PROTOCOL)

    if process_flow['update_posteriors']:
        print('update_posteriors...')
        with open(results_path + 'ltv_class.pkl', 'rb') as input:
            the_class = pickle.load(input)
        the_class.get_reports_data(report_filenames)
        the_class.get_users(max_users=-1)
        the_class.update_posteriors()

    if process_flow['calc_ltv']:
        print('calc ltv...')
        the_class.get_ltv()

    if process_flow['update_ranking']:
        print('update_ranking...')
        with open(results_path + 'ltv_class.pkl', 'rb') as input:
            the_class = pickle.load(input)
        the_class.get_reports_data(report_filenames)
        the_class.get_users(max_users=-1)
        the_class.update_rankings()
        the_class.get_ltv_ranking()

    if process_flow['check_result']:
        print('check result...')
        the_class.check_result(present_filename)

        print('saving data...')
        with open(results_path + c_type + '_final_results.csv', 'w') as csv_file:
            str_title = 'user_id, '

            str_title += 'week_predicted, week_ground_truth, week_error, week_error_sq, week_correct_0, '
            str_title += 'month_predicted, month_ground_truth, month_error, month_error_sq, month_correct_0, '
            str_title += '3months_predicted, 3months_ground_truth, 3months_error, 3months_error_sq, 3months_correct_0'

            csv_file.write(str_title + '\n')
            for ids, d in the_class.final_results.items():
                csv_line = ids
                for i in range(3):
                    csv_line += ',' + str(d['predicted'][i])
                    csv_line += ',' + str(d['ground_truth'][i])
                    csv_line += ',' + str(d['predicted'][i] - d['ground_truth'][i])
                    csv_line += ',' + str(np.square(d['predicted'][i] - d['ground_truth'][i]))
                    if (d['predicted'][i] == 0 and d['ground_truth'][i] == 0) or \
                        (d['predicted'][i] > 0 and d['ground_truth'][i] > 0):
                        csv_line += ', 1'
                    else:
                        csv_line += ', 0'
                csv_file.write(csv_line + '\n')