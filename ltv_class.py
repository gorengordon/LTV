import csv
import numpy as np
import copy


def get_number_of_csv_lines(filename):
    num_lines = -1
    with open(filename, 'rb') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        for row in file_reader:
            num_lines += 1
    return num_lines

data_path = '../../LTV/data/'
period_names = {'week': 7, 'month': 30}


class LtvClass:

    def __init__(self, filename, max_count=1000, target_periods=None):
        self.filename = filename
        self.max_count = max_count
        self.n_bins = 20  # number of bins in distribution

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
            self.priors.append(np.zeros([self.n_bins]))

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
                            self.priors[i][v_bin] += 1.0 / self.value_stats[i]['count']
                        except:
                            pass
                row_count += 1
                if row_count >= self.max_count:
                    break

    def get_reports_data(self, report_filenames):
        self.report_data = []
        for rf in report_filenames:
            s = rf.split('_')
            period_index = self.target_periods.index(period_names[s[2]])
            more_less = s[1][:4] == 'more'
            report_value = float(s[1][4:])
            self.report_data.append({
                'filename': rf,
                'type': s[0],
                'period_index': period_index,
                'more_less': more_less,
                'report_value': report_value,
            })

    def get_users(self):
        self.user_data = {}
        for rd in self.report_data:
            with open(data_path + 'reports/' + rd['filename'] + '.csv', 'rb') as csv_file:
                report_file = csv.reader(csv_file, delimiter=',')
                for row in report_file:
                    user_id = row[0]
                    if user_id not in self.user_data:
                        self.user_data[user_id] = {
                            'prob': copy.deepcopy(self.priors)
                        }

    def update_posteriors(self):
        for rd in self.report_data:
            with open(data_path + 'reports/' + rd['filename'] + '.csv', 'rb') as csv_file:
                report_file = csv.reader(csv_file, delimiter=',')
                for row in report_file:
                    user_id = row[0]
                    the_prob = 0.5 #float(row[2])  # missing ------------------------------------------------------
                    self.the_update(user_id, the_prob, rd)

    def the_update(self, user_id, the_prob, rd):
        this_prior = self.user_data[user_id]['prob'][rd['period_index']]
        report_value = rd['report_value']
        report_value_bin = self.convert_value_to_bin(report_value, self.value_stats[rd['period_index']])

        # Bayes
        the_posterior = copy.deepcopy(this_prior)
        for p in the_posterior:
            p = p +1 # CHANGE

        self.user_data[user_id]['prob'][rd['period_index']] = the_posterior
