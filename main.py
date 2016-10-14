from ltv_class import *
import pickle

process_flow = {
    'get_stats': True,
    'get_priors': True,
    'update_posteriors': True,
    'calc_ltv': True
}

filename = data_path + 'Ground_truth/ref-20160401.csv'
results_path = '../../LTV/results/'

if process_flow['get_stats']:
    print('get_stats...')
    the_class = LtvClass(filename=filename, target_periods=[7,30,90], max_count=10000, max_users=10000)
    the_class.get_stats()
    with open(results_path + 'ltv_class.pkl', 'wb') as output:
        pickle.dump(the_class, output, pickle.HIGHEST_PROTOCOL)

if process_flow['get_priors']:
    print('get_priors...')
    with open(results_path + 'ltv_class.pkl', 'rb') as input:
        the_class = pickle.load(input)
    the_class.define_priors(n_bins=20)
    the_class.update_priors()
    with open(results_path + 'ltv_class.pkl', 'wb') as output:
        pickle.dump(the_class, output, pickle.HIGHEST_PROTOCOL)

if process_flow['update_posteriors']:
    print('update_posteriors...')
    with open(results_path + 'ltv_class.pkl', 'rb') as input:
        the_class = pickle.load(input)
    the_class.get_reports_data(['FTBs_less10_week'])
    the_class.get_users()
    the_class.update_posteriors()
    with open(results_path + 'ltv_class.pkl', 'wb') as output:
        pickle.dump(the_class, output, pickle.HIGHEST_PROTOCOL)

if process_flow['calc_ltv']:
    print('calc ltv...')
    with open(results_path + 'ltv_class.pkl', 'rb') as input:
        the_class = pickle.load(input)
    the_class.get_ltv()
    with open(results_path + 'ltv_class.pkl', 'wb') as output:
        pickle.dump(the_class, output, pickle.HIGHEST_PROTOCOL)

print('loading results...')
with open(results_path + 'ltv_class.pkl', 'rb') as input:
    the_class = pickle.load(input)
print('done!')
for u in the_class.user_data.values():
    print(u['ltv'])

# for p in the_class.priors:
#     print(np.log(p))
#
# print(the_class.target_periods)
# generate prior distribution

# go over reports
# update per ID the posterior per report