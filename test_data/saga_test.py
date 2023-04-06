import saga 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = saga.load_csv("../FED3V1_FP_test.csv")
events_df = saga.load_events(
    "FED0 timestamps2023-03-15T10_42_22.csv",
    "FED1 timestamps2023-03-15T10_42_22.csv"
)



frames = saga.deinterleive(dataframe, "LedState", [1,2,4])
#Assumes LED_STATE == 1 is the reference signal


min_size = saga.smallest_size(frames)
frames = list(map(lambda df: saga.truncate(df,min_size), frames))

SIGNALS = {
    "Region0G":[0,"Green_Left"], 
    "Region2R":[1,"Red_Left"], 
    "Region1G":[0,"Green_Right"], 
    "Region3R":[1,"Red_Right"],
}

# Region Column | LED state index (starting at 1) | Name

frames = list(
    map(
        lambda df: saga.smooth(df, SIGNALS.keys(), 20),
        frames,
    )
)

frames = list(
     map(lambda df: saga.correct_photobleach(df, "Timestamp", SIGNALS.keys()), frames)
)



#Corrects the data by the bi_exonential fit
frames = list(
    map(lambda df: saga.correct_reference(df, frames[0], SIGNALS.keys()), frames[1:])
)

frames = list(
    map(
        lambda df: saga.label_events(df,events_df),
        frames
    )
)

##events 

DURATION = 60
event_signals = saga.split_events(frames, DURATION, "EVENT_FLAG", "FED_ID", "Timestamp", SIGNALS)

## Normalize

for signal in event_signals:
    for fed in event_signals[signal]:
        for event_type in event_signals[signal][fed]:
            event_signals[signal][fed][event_type] = saga.normalize_df(event_signals[signal][fed][event_type])

## Collect_Data


for signal in event_signals:
    for fed in event_signals[signal]:
        for event_type in event_signals[signal][fed]:
            events = event_signals[signal][fed][event_type]
            event_signals[signal][fed][event_type] = pd.DataFrame([
                event_data.to_numpy() for event_data in events
            ]).T


## SEM

SEMs = {}

for signal in event_signals:
    SEMs[signal] = {}
    for fed in event_signals[signal]:
        SEMs[signal][fed] = {}
        for event_type in event_signals[signal][fed]:
            SEMs[signal][fed][event_type] = saga.signal_sems(event_signals[signal][fed][event_type])


Signal = SEMs['Red_Right'][0.0][1]

time_stamps = np.linspace(0, DURATION, Signal.shape[0])

plt.plot(time_stamps, Signal.Mean, alpha=0.5)
plt.fill_between(
    time_stamps,
    Signal.Mean - Signal.SEM,
    Signal.Mean + Signal.SEM,
    alpha=0.2
)
plt.show()



