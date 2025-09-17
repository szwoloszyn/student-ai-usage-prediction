import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
DATASET = "dataset/ai_assistant_usage_student_life.csv"

def load_ai_usage_data():
    if not Path(DATASET).is_file():
        raise TypeError
    return pd.read_csv(Path("dataset/ai_assistant_usage_student_life.csv"))

data = load_ai_usage_data()
# print(data['SessionLengthMin'])
# print(data.head())

## dropping SessionID and modifying SessionDate to days since oldest entry
data['SessionDate'] = pd.to_datetime(data['SessionDate'])

oldest_date = pd.to_datetime(datetime.datetime.now())
for date in data['SessionDate']:
    if date >= oldest_date:
        continue
    oldest_date = date
data['DaysSinceFirstSession'] = (data['SessionDate'] - oldest_date).dt.days

## modyfying FinalOutcome and StudentLevel to number
final_outcome_translator = {
    'Assignment Completed' : 3,
    'Idea Drafted' : 2,
    'Confused' : 1,
    'Gave Up' : 0,
}
data['FinalOutcome'] = [final_outcome_translator[entry] for entry in data['FinalOutcome']]

# print(data.tail(8))
print(data['StudentLevel'].unique())

## v. alpha is gonna exclude remaining categorial features - TaskType, Discipline and StudentLevel
X = data.drop(columns=["UsedAgain", "SessionID", "SessionDate", "Discipline", "TaskType", "StudentLevel"])
y = data["UsedAgain"].astype(int)  # True/False -> 1/0
print( X.tail(10) )

### looking at correlation of each feature

# TODO

# ##! percentage of students returning to AI depending on their SatisfactionRating - not sufficient

# used_again_pct = (
#     data.groupby('SatisfactionRating')['UsedAgain']
#       .mean() * 100
# )
# plt.plot(used_again_pct)

# 

# ##! many test plots

sampled_data = data.sample(n=500, random_state=42)

sns.lmplot(
    data=sampled_data,
    x="AI_AssistanceLevel",
    y="DaysSinceFirstSession",
    col="StudentLevel",
      hue="UsedAgain",
    palette={True: "blue", False: "red"},
    fit_reg=False
)
plt.show()
