import csv
import numpy as np
import copy
from operator import mul
import re

period_names = {'week': 7, 'month': 30, '3months': 90}
data_path = '../../LTV/data/'
reports_path = '../../LTV/data/ltv/'


def get_number_of_csv_lines(filename):
    num_lines = -1
    with open(filename, 'rb') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        for row in file_reader:
            num_lines += 1
    return num_lines


class LtvClass:

    def __init__(self, filename, max_count=1000, target_periods=None):
        self.filename = filename
        self.max_count = max_count
        self.max_users = 1000
        self.n_bins = 100  # number of bins in distribution

        self.prob = {}  # prob: key=id, value = []_num_target_periods = numpy p(amount)
        self.priors = None  # numpy[t, a]
        # read past ground truth
        self.num_values = 0

        self.data_target_periods = []
        if target_periods is None:
            self.num_target_periods = 0
            self.target_periods = []  # number of days to predict to
        else:
            self.num_target_periods = len(target_periods)
            self.target_periods = target_periods
        self.value_stats = []
        self.report_data = []

        self.user_data = {}

    def get_stats(self):
        with open(self.filename, 'rb') as csv_file:
            file_reader = csv.reader(csv_file, delimiter=',')
            row_count = 0
            for row in file_reader:
                if row_count == 0:
                    for col in row:
                        s = col.split('_')
                        if len(s) == 3:
                            self.data_target_periods.append(int(s[2]))

                    if not self.target_periods:
                        self.target_periods = self.data_target_periods
                    self.num_target_periods = len(self.target_periods)

                    for i in range(self.num_target_periods):
                        self.value_stats.append({
                            'min': np.nan,
                            'max': np.nan,
                            'count': 0,
                            'sum': 0,
                            'mean': np.nan
                        })

                else:
                    values = [float(x) for x in row][1:]
                    for e, v in enumerate(values):
                        try:
                            i = self.target_periods.index(self.data_target_periods[e])
                            self.value_stats[i]['sum'] += v
                            self.value_stats[i]['count'] += 1
                            self.value_stats[i]['min'] = np.nanmin([self.value_stats[i]['min'], v])
                            self.value_stats[i]['max'] = np.nanmax([self.value_stats[i]['max'], v])
                            self.value_stats[i]['mean'] = float(self.value_stats[i]['sum']) / float(self.value_stats[i]['count'])
                        except:
                            pass
                row_count += 1
                if row_count >= self.max_count:
                    break

    def define_priors(self, n_bins=20):
        self.n_bins = n_bins
        self.priors = []
        for i in range(self.num_target_periods):
            minimum_prior = 1.0 / float(self.value_stats[i]['count'] + self.n_bins)
            self.priors.append(np.ones([self.n_bins]) * minimum_prior)

    def convert_value_to_bin(self, value, value_stats):
        # log-distribution
        the_value = np.log(value+1)
        min_value = np.log(value_stats['min']+1)
        max_value = np.log(value_stats['max']+1)
        the_bin = int(np.floor(float(self.n_bins-1.0) *
            (the_value - min_value) /
            (max_value - min_value)
        ))
        return max([min([the_bin, self.n_bins-1]), 0])

    def convert_bin_to_value(self, b, value_stats):
        # log-distribution
        min_value = np.log(value_stats['min'] + 1)
        max_value = np.log(value_stats['max'] + 1)
        log_value = min_value + (max_value - min_value) * float(b) / float(self.n_bins - 1.0)
        return np.exp(log_value) - 1.0

    def update_priors(self):
        with open(self.filename, 'rb') as csv_file:
            file_reader = csv.reader(csv_file, delimiter=',')
            row_count = 0
            for row in file_reader:
                if row_count > 0:
                    values = [float(x) for x in row][1:]
                    for e, v in enumerate(values):
                        try:
                            i = self.target_periods.index(self.data_target_periods[e])
                            v_bin = self.convert_value_to_bin(v, self.value_stats[i])
                            self.priors[i][v_bin] += 1.0 / float(self.value_stats[i]['count'] + self.n_bins)
                        except:
                            pass
                row_count += 1
                if row_count >= self.max_count:
                    break
        for i, p in enumerate(self.priors):
            p_value = self.get_value_from_prob(p,i)

    # ========= probability ============================

    def get_reports_data(self, report_filenames):
        self.report_data = []
        for rf in report_filenames:
            m = re.search('coming(.+?).csv', rf)
            period_index = self.target_periods.index(period_names[m.group(1)])
            more_less = 'more' in rf
            m = re.search('than(.+?)coming', rf)
            report_value = float(m.group(1))
            FTB_Users = 'FTB' in rf

            prob_less_in_prior = self.get_prob_of_less_than_value(period_index, self.priors[period_index], report_value)
            if more_less:
                query_prob = 1 - prob_less_in_prior
            else:
                query_prob = prob_less_in_prior
            self.report_data.append({
                'filename': rf,
                'FTB_Users': FTB_Users,
                'period_index': period_index,
                'more_less': more_less,
                'report_value': report_value,
                'prob_in_prior': query_prob
            })

    def get_users(self, max_users=1000):
        self.max_users = max_users
        self.user_data = {}
        for rd in self.report_data:
            with open(reports_path + rd['filename'], 'rb') as csv_file:
                report_file = csv.reader(csv_file, delimiter=',')
                for row in report_file:
                    user_id = row[0]
                    if user_id not in self.user_data:
                        self.user_data[user_id] = {
                            'prob': copy.deepcopy(self.priors),
                            'ltv': np.zeros([self.num_target_periods]),
                            'rank_prediction': []
                        }
                        for i in self.target_periods:
                            self.user_data[user_id]['rank_prediction'].append([])

                        if self.max_users > 0 and len(self.user_data) > self.max_users:
                            return
        print('users: ', len(self.user_data))

    def update_posteriors(self):
        for rd in self.report_data:
            with open(reports_path + rd['filename'], 'rb') as csv_file:
                print('processing ', rd['filename'], '...')
                report_file = csv.reader(csv_file, delimiter=',')
                row_counter = 0
                errors = 0
                for row in report_file:
                    if row_counter > 0:
                        user_id = row[0]
                        the_prob = float(row[2])

                        if not self.the_update(user_id, the_prob, rd):
                            errors += 1
                    row_counter += 1
                print('done ', row_counter, '(', errors, ' errors)')

    def the_update(self, user_id, the_prob, rd):
        # try:
        this_prior = self.user_data[user_id]['prob'][rd['period_index']]
        # except:
            # print('the user: ', user_id, ' is not in the first ', self.max_users, ' users found')
            # return

        report_value = rd['report_value']
        report_value_bin = self.convert_value_to_bin(report_value, self.value_stats[rd['period_index']])

        prior_mean_ltv = self.get_value_from_prob(this_prior, rd['period_index'])

        # Bayes
        the_posterior = copy.deepcopy(this_prior)
        for b, p in enumerate(the_posterior):
            if rd['more_less']:
                if b > report_value_bin:
                    the_posterior[b] *= the_prob / rd['prob_in_prior']
                else:
                    the_posterior[b] *= (1 - the_prob) / (1 - rd['prob_in_prior'])
            else:
                if b < report_value_bin:
                    the_posterior[b] *= the_prob / rd['prob_in_prior']
                else:
                    the_posterior[b] *= (1 - the_prob) / (1 - rd['prob_in_prior'])

        posterior_mean_ltv = self.get_value_from_prob(the_posterior, rd['period_index'])
        print('mean ltv:', prior_mean_ltv, posterior_mean_ltv)

        # sanity checks!
        sanity_checks = True
        if LtvClass.get_sum_prob(this_prior) < 0.999:
            # print('Error: prior does not sum to one: ', LtvClass.get_sum_prob(this_prior))
            sanity_checks = False
        if LtvClass.get_sum_prob(the_posterior) < 0.999:
            # print('Error: posterior does not sum to one', LtvClass.get_sum_prob(this_prior), LtvClass.get_sum_prob(the_posterior))
            sanity_checks = False
        if LtvClass.get_sum_prob(np.abs(np.array(the_posterior) - np.array(this_prior))) < 0.001:
            # print('Error: posterior did not update', LtvClass.get_sum_prob(np.abs(np.array(the_posterior) - np.array(this_prior))))
            sanity_checks = False

        sum_posterior = LtvClass.get_sum_prob(the_posterior)
        for b in range(len(the_posterior)):
            the_posterior[b] /= sum_posterior

        self.user_data[user_id]['prob'][rd['period_index']] = copy.deepcopy(the_posterior)
        return sanity_checks

    def get_prob_of_less_than_value(self, period_index, prior, value):
        value_bin = self.convert_value_to_bin(value, self.value_stats[period_index])
        prob_list = [prior[x] for x in range(value_bin)]
        return np.sum(prob_list)

    def get_value_from_prob(self, prob, period_index):
        value = 0
        for b, p in enumerate(prob):
            value += p * self.convert_bin_to_value(b, self.value_stats[period_index])
        return value

    def get_ltv(self):
        self.final_results = {}
        for id, u in self.user_data.items():
            self.final_results[id] = {'predicted': [], 'ground_truth': []}
            for i, prob in enumerate(u['prob']):
                self.final_results[id]['predicted'].append(self.get_value_from_prob(prob, i))
                self.final_results[id]['ground_truth'].append(-1)

    @staticmethod
    def get_sum_prob(prob):
        sum_prob = np.sum(prob)
        return sum_prob

    def check_result(self, filename):
        with open(self.filename, 'rb') as csv_file:
            file_reader = csv.reader(csv_file, delimiter=',')
            row_count = 0
            for row in file_reader:
                if row_count > 0:
                    user_id = row[0]
                    values = [float(x) for x in row][1:]
                    for e, v in enumerate(values):
                        try:
                            i = self.target_periods.index(self.data_target_periods[e])
                            self.final_results[user_id]['ground_truth'][i] = v
                        except:
                            pass
                row_count += 1

    # ========= ranking ========================

    def update_rankings(self):
        for rd in self.report_data:
            with open(reports_path + rd['filename'], 'rb') as csv_file:
                report_file = csv.reader(csv_file, delimiter=',')
                candidate_count = 0
                for row in report_file:
                    candidate_count += 1

            cumulative_candidate = [0]
            for p in self.priors[rd['period_index']]:
                cumulative_candidate.append(cumulative_candidate[-1] + int(p * candidate_count))

            with open(reports_path + rd['filename'], 'rb') as csv_file:
                print('processing ', rd['filename'], '...')
                report_file = csv.reader(csv_file, delimiter=',')
                row_counter = 0
                for row in report_file:
                    if row_counter > 0:
                        user_id = row[0]
                        the_rank = float(row[1])

                        self.the_update_rank(user_id, the_rank, cumulative_candidate, rd)
                    row_counter += 1

    def the_update_rank(self, user_id, the_rank, cumulative_candidate, rd):
        for rank_bin, c in enumerate(cumulative_candidate):
            if rank_bin > 0:
                if rd['more_less']:     # if more, rank=1 --> last bin, rank=candidate_count --> first bin
                    if the_rank > c:
                        break
                else:                   # if less, rank=1 --> first bin, rank=candidate_count --> last bin
                    if the_rank < c:
                        break
        rank_bin -= 1

        rank_based_value = self.convert_bin_to_value(rank_bin, self.value_stats[rd['period_index']])
        self.user_data[user_id]['rank_prediction'][rd['period_index']].append(int(rank_based_value))

    def get_ltv_ranking(self):
        self.final_results = {}
        for id, u in self.user_data.items():
            self.final_results[id] = {'predicted': [], 'ground_truth': []}
            for i, rank_prediction in enumerate(u['rank_prediction']):
                self.final_results[id]['predicted'].append(np.median(rank_prediction))
                self.final_results[id]['ground_truth'].append(-1)