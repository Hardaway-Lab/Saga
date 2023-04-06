import numpy as np
import pandas as pd
import bara
import matplotlib.pyplot as plt

def gen_pulse(max_pulse, pulse_length, pulse_count):
    out = []
    pulses = np.random.randint(0,max_pulse,pulse_count)
    print(pulses)
    out.extend(np.zeros(np.random.randint(max_pulse*pulse_length*1.5,3*max_pulse*pulse_length)))
    for pulse in pulses:
        for i in range(pulse):
            out.extend([1]*pulse_length)
            out.extend([0]*pulse_length)
        out.extend(np.zeros(np.random.randint(max_pulse*pulse_length,2*max_pulse*pulse_length)))
    return out


max_pulses = 5
pulse_size = 4
pulse_counts = 10

pulse = gen_pulse(max_pulses,pulse_size,pulse_counts)

df = pd.DataFrame({"pulse":pulse})
plt.plot(df.pulse)
bara.ttl_edge_count(df,"pulse",max_pulses*pulse_size*4//3)

plt.plot(df.pulse)
plt.show()
