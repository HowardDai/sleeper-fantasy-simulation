import pandas as pd
import random
import copy
import urllib.request
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TKAgg')

league_id = "1257435321281286144"

#get owners, flatten metadata, and extract user_id, roster_id, and team_name
with urllib.request.urlopen("https://api.sleeper.app/v1/league/" + league_id + "/users") as url:
    owners = json.loads(url.read().decode())
for owner in range(len(owners)):
	for datapoint in owners[owner]["metadata"]:
		owners[owner][datapoint] = owners[owner]["metadata"][datapoint]
	del owners[owner]["metadata"]

owners_df = pd.DataFrame(owners)[["user_id", "display_name", "team_name"]]

owners_df['team_name'] = owners_df['team_name'].replace('', pd.NA)
owners_df['team_name'] = owners_df['team_name'].fillna(owners_df['display_name'])


DIVISION_1_IDS = ["635385669602095104", "996314837284225024", "873068475000782848", "996332115534958592", "1132002915557859328", "1134218267205193728"]

divisions_df = owners_df[["user_id"]].copy()
divisions_df["division"] = divisions_df["user_id"].apply(
    lambda uid: 1 if uid in DIVISION_1_IDS else 2
)
print(divisions_df)
#get rosters, flatten settings, and extract owner_id, roster_id, and wins
with urllib.request.urlopen("https://api.sleeper.app/v1/league/" + league_id + "/rosters") as url:
    rosters = json.loads(url.read().decode())
for roster in range(len(rosters)):
	for datapoint in rosters[roster]["settings"]:
		if datapoint == 'wins':
			rosters[roster][datapoint] = rosters[roster]["settings"][datapoint]
	del rosters[roster]["settings"]
rosters_df = pd.DataFrame(rosters)[["owner_id", "roster_id", "wins"]]


#merge owners and rosters
owners_df = pd.merge(owners_df, rosters_df, left_on="user_id", right_on="owner_id")[['user_id', 'display_name', 'roster_id', 'team_name']]

#get matchups and extract roster_id, matchup_id, points, and week, and also find upcoming gameweek
upcoming_gameweek = 13
matchups_df = pd.DataFrame()
for week in range(1, 15):
	with urllib.request.urlopen(("https://api.sleeper.app/v1/league/" + league_id + "/matchups/" + str(week))) as url:
		matchups = json.loads(url.read().decode())

	temporary_df = pd.DataFrame(matchups)[['roster_id', 'matchup_id', 'points']]
	temporary_df['week'] = week
	matchups_df = pd.concat([matchups_df, temporary_df], ignore_index=True)
	if temporary_df['points'].sum() != 0:
		upcoming_gameweek = week + 1

NUM_PAST_WEEKS = 12

lambda_ = 0.9
weights = np.array([lambda_**i for i in range(12)][::-1])  # week 1 -> 12
# weights = np.ones(12)
print("WEIGHTS", weights)
wm = lambda x: np.average(x, weights=weights)


#get team averages and std deviations, then merge with owners to get user_id
team_stats_df = matchups_df[matchups_df['points'] != 0][matchups_df['week'] >= upcoming_gameweek - NUM_PAST_WEEKS][['roster_id', 'points']].groupby('roster_id').agg([wm, 'std']).reset_index()
team_stats_df.columns = ['roster_id', 'mean_points', 'std_points']
team_stats_df = pd.merge(team_stats_df, owners_df, on='roster_id')[['user_id', 'team_name', 'mean_points', 'std_points']]

team_stats_df_2 = matchups_df[matchups_df['points'] != 0][['roster_id', 'points']].groupby('roster_id').agg(['sum']).reset_index()
team_stats_df_2.columns = ['roster_id', 'sum_points']
team_stats_df_2 = pd.merge(team_stats_df_2, owners_df, on='roster_id')[['user_id', 'sum_points']]

#merge owners with matchups to get user_ids instead of roster_ids for both teams
matchups_df = pd.merge(matchups_df, owners_df, on='roster_id')[['user_id', 'display_name', 'matchup_id', 'points', 'week']]
matchups_df = pd.merge(matchups_df, matchups_df, on=('week', 'matchup_id'))
matchups_df = matchups_df[matchups_df['user_id_x'] != matchups_df['user_id_y']].drop_duplicates(['week', 'matchup_id'])

#select remaining matchups
remaining_matchups_df = matchups_df[matchups_df['week'] >= upcoming_gameweek]

# print("REMAINING MATCHUPS DF", remaining_matchups_df['week'])


"""
simulate an individual matchup
return each team's points in dict
"""

print("SAVING TEAM STATS TO FILE")
team_stats_df.to_csv("team_stats.csv")



def matchup_simulator(team_stats_df, matchup):
	team_x = matchup['user_id_x']
	team_y = matchup['user_id_y']

	#get team_x points
	team_x_mean = team_stats_df[team_stats_df['user_id'] == team_x]['mean_points'].values
	team_x_std = team_stats_df[team_stats_df['user_id'] == team_x]['std_points'].values
	team_x_points = random.normalvariate(team_x_mean, team_x_std)

	#get team_y points
	team_y_mean = team_stats_df[team_stats_df['user_id'] == team_y]['mean_points'].values
	team_y_std = team_stats_df[team_stats_df['user_id'] == team_y]['std_points'].values
	team_y_points = random.normalvariate(team_y_mean, team_y_std)

	return {team_x: team_x_points, team_y: team_y_points}	

#construct team_dict
team_dict = {}
for index, team in owners_df.iterrows():
	team_dict[team['user_id']] = 0

"""
run n trials of rest-of-season simulation
return seed_pivot and team_dict  
"""
def trial_runner(n, team_dict):

	#get current team wins
	original_team_wins_dict = {}
	for index, team in rosters_df.iterrows():
		original_team_wins_dict[team['owner_id']] = team['wins']
  
	original_team_points_dict = {}
 
	

	for index, team in team_stats_df_2.iterrows():
		original_team_points_dict[team['user_id']] = team['sum_points'] 
  
	# print("ORIG TEAM WINS DICT", original_team_wins_dict)
	# print("ORIG TEAM POINTS DICT", original_team_points_dict)


	#create df for overall seeding - each team mapped to each possible seed
	total_seed_df = pd.DataFrame(team_dict.items(), columns=['user_id', 'occurences'])
	total_seed_df['key'] = 0
	possible_seeds_df = pd.DataFrame({'seed': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
	possible_seeds_df['key'] = 0
	total_seed_df = pd.merge(total_seed_df, possible_seeds_df, on='key')[['user_id', 'seed', 'occurences']]


	#run n trials
	for i in range(n):
		if i % 100 == 0:
			print(i)
	
		seeding_dict = {}
		team_wins_dict = original_team_wins_dict.copy()
		team_points_dict = original_team_points_dict.copy()

		#run through each week remaining in season
		for week in list(remaining_matchups_df['week'].unique()):
			points_dict = {}

			#run matchup_simulator on all remaining matchups in a week
			for index, matchup in remaining_matchups_df[remaining_matchups_df['week'] == week].iterrows():
				matchup_points_dict = matchup_simulator(team_stats_df, matchup)
				points_dict.update(matchup_points_dict)

				#award head-to-head win to winning team
				if points_dict[matchup['user_id_x']] >= points_dict[matchup['user_id_y']]:
					team_wins_dict[matchup['user_id_x']] += 1
					
				else:
					team_wins_dict[matchup['user_id_y']] += 1
				team_points_dict[matchup['user_id_x']] += points_dict[matchup['user_id_x']]
				team_points_dict[matchup['user_id_y']] += points_dict[matchup['user_id_y']]

		#determine playoffs
		team_wins_df = pd.DataFrame.from_dict(team_wins_dict, orient="index", columns=['wins'])
		team_points_df = pd.DataFrame.from_dict(team_points_dict, orient="index", columns=['points'])
		team_wins_df = pd.merge(team_wins_df, team_points_df, left_index=True, right_index=True, how='inner')
  
		playoff_teams_df = team_wins_df.sort_values(by= ['wins', 'points'], ascending=False)
		playoff_teams_df = pd.merge(playoff_teams_df, divisions_df, left_index=True, right_on="user_id")
		playoff_teams_df = playoff_teams_df.set_index("user_id", drop=True)
		
		trial_team_wins_df = pd.DataFrame.from_dict(team_wins_dict, orient="index", columns=['wins'])
		trial_team_wins_df['seed'] = team_wins_df['wins'].rank(method='first', ascending=False)
		# Modifying seedings, custom rules for our league 

		# Helper: flip division (assuming only 1 and 2)
		def other_division(div):
			return 1 if div == 2 else 2

		# Start with everyone
		remaining = playoff_teams_df.copy()
		seeds = [None] * 12    # seeds[0] = 1-seed, ..., seeds[11] = 12-seed

		# 1 SEED: Highest overall
		remaining = remaining.sort_values(by=['wins', 'points'], ascending=False)
		seed1 = remaining.index[0]
		seeds[0] = seed1
		seed1_div = remaining.loc[seed1, 'division']
		remaining = remaining.drop(seed1)

		# 2 SEED: highest in the other division (relative to seed 1)
		cand = remaining[remaining['division'] == other_division(seed1_div)].sort_values(
			by=['wins', 'points'], ascending=False
		)
		seed2 = cand.index[0]
		seeds[1] = seed2
		remaining = remaining.drop(seed2)

		# 3 SEED: Next highest overall
		remaining = remaining.sort_values(by=['wins', 'points'], ascending=False)
		seed3 = remaining.index[0]
		seeds[2] = seed3
		seed3_div = remaining.loc[seed3, 'division']
		remaining = remaining.drop(seed3)

		# 4 SEED: next highest in the other division (relative to seed 3)
		cand = remaining[remaining['division'] == other_division(seed3_div)].sort_values(
			by=['wins', 'points'], ascending=False
		)
		seed4 = cand.index[0]
		seeds[3] = seed4
		remaining = remaining.drop(seed4)

		# 5 SEED: Next highest overall
		remaining = remaining.sort_values(by=['wins', 'points'], ascending=False)
		seed5 = remaining.index[0]
		seeds[4] = seed5
		remaining = remaining.drop(seed5)

		# 6 SEED: Next highest points (wins as tiebreaker)
		seed6 = remaining.sort_values(by=['points', 'wins'], ascending=False).index[0]
		seeds[5] = seed6
		remaining = remaining.drop(seed6)

		# At this point, 6 teams left in `remaining`

		# 12 SEED: Worst overall
		remaining = remaining.sort_values(by=['wins', 'points'], ascending=[True, True])
		seed12 = remaining.index[0]
		seeds[11] = seed12
		seed12_div = remaining.loc[seed12, 'division']
		remaining = remaining.drop(seed12)

		# 11 SEED: Worst overall in the other division (relative to seed 12)
		cand = remaining[remaining['division'] == other_division(seed12_div)].sort_values(
			by=['wins', 'points'], ascending=[True, True]
		)
		seed11 = cand.index[0]
		seeds[10] = seed11
		remaining = remaining.drop(seed11)

		# Now 4 teams left → SEEDS 7–10: rest by worst→best overall
		remaining = remaining.sort_values(by=['wins', 'points'], ascending=[False, False])
		for offset, team_id in enumerate(remaining.index):
			seeds[6 + offset] = team_id  # fills seeds[6], [7], [8], [9]

		# `seeds` now contains user_ids for seeds 1–12
		for idx, user_id in enumerate(seeds):
			playoff_teams_df.loc[user_id, 'seed'] = idx + 1
		# print("TRIAL TEAM WINS")
		# print(playoff_teams_df)
		trial_team_wins_df = playoff_teams_df

 
		for trial_user_id, trial_row in trial_team_wins_df.iterrows():
			for index, total_row in total_seed_df.iterrows():
				if trial_user_id == total_row['user_id']:
					if int(trial_row['seed']) == int(total_row['seed']):
						total_seed_df.loc[index, 'occurences'] += 1

	#get team name from owners_df, pivot, and add Playoff Probability column (sum of top 6 seeds)
	total_seed_df = pd.merge(total_seed_df, owners_df, on='user_id')[['team_name', 'seed', 'occurences']]
	print("total_seed_df:\n")
	print(total_seed_df)


	seed_pivot = pd.pivot_table(total_seed_df, values='occurences', index='team_name', columns='seed', aggfunc=lambda x:12*x.sum()/total_seed_df['occurences'].sum())
	seed_pivot['Playoff Probability'] = seed_pivot[[1,2,3,4,5,6]].sum(axis=1)
	seed_pivot = seed_pivot.sort_values(['Playoff Probability', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ascending=False)
	return (seed_pivot, team_dict)

n = 1000
seed_pivot, team_dict = trial_runner(n, team_dict)
print(seed_pivot)
seed_pivot.to_csv("playoff_probabilities.csv")
#create df to exclude last column from heatmap coloring
heatmap_colors = seed_pivot.iloc[:,:12]
heatmap_colors['Playoffs'] = 0

annot_data = seed_pivot.round(2)


#create and show heatmap
ax = sns.heatmap(heatmap_colors, annot=annot_data, alpha=1, cmap='Greens', linewidths=1, cbar=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_xlabel('Seed')
ax.set_ylabel('Team')
plt.title(('Seed Probability, n = ' + str(n) + ' Trials'))
plt.tight_layout()
plt.show()

#add results to log
team_df = pd.DataFrame(team_dict.items(), columns=['user_id', 'successes'])
team_df = pd.merge(team_df, owners_df)[['display_name', 'successes']].sort_values('successes', ascending=False)
team_df['week'] = upcoming_gameweek
team_df['success %'] = team_df['successes'] / n 
team_df.to_csv("WeeklySeedProbabilityLog.csv", mode='a', header=False)

	







