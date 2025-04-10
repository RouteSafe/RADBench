import json
import os
from sklearn.metrics import roc_auc_score
import torch
import pickle
from BGP_Anomaly_detection.Self_Attention_LSTM import SA_LSTM
from dataloader import SingleEventDataset
from torch.utils.data import DataLoader
from commons import Metric, read_event_list

# Metric("").calculate_point_wise(result_path="mydata_test_result")
# exit(0)
events = read_event_list(evnet_type='leak')
all_events_result = []
loose_events_result = []
for event in events:
    event_name = event['event_name']
    event_start_time = event['event_start_time']
    event_end_time = event['event_end_time']
    leaker_as = event['leaker_as']
    print(f'processing event:{event_name}')
    dataset = SingleEventDataset(
        event_path=f"/data/data/anomaly-event-routedata/{event_name}",
        event_start_time=event_start_time,
        event_end_time=event_end_time,
        victim_AS=leaker_as,
        # force_reCalc=True
    )
    BATCH_SIZE = len(dataset)
    test_loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

    Path = 'params/best_lstm_params_route_leak.pkl'
    model = SA_LSTM(WINDOW_SIZE=30, INPUT_SIZE=83, Hidden_SIZE=128,
                     LSTM_layer_NUM=1)
    model.load_state_dict(torch.load(Path))
    with torch.no_grad():
        for feature, label in test_loader:
            test_output, attn = model(feature)

            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()

            metric = Metric(event_name)
            (TN, FP, FN, TP), report, precision, recall, F1_score = \
                metric.calculate_metrics(label.data.numpy().tolist(), pred_y.tolist())
            
            auc = Metric._calc_auc(label.data.numpy().tolist(), pred_y.tolist())
            if auc<0.5:
                all_events_result.append(0)
            else:
                all_events_result.append(1)
            if recall>0:
                loose_events_result.append(1)
            else:
                loose_events_result.append(0)

print(f'Event-Wise: {all_events_result.count(1)}/{len(all_events_result)}')
print(f'loose Event-Wise: {loose_events_result.count(1)}/{len(loose_events_result)}')
with open(os.path.join('mydata_test_result', 'event-wise metric.txt'), 'w') as f:
    f.write(str(all_events_result))
    f.write(f'\nEvent-Wise: {all_events_result.count(1)}/{len(all_events_result)}')
with open(os.path.join('mydata_test_result', 'event-wise metric(all anomaly).txt'), 'w') as f:
    f.write(str(loose_events_result))
    f.write(f'\nloose Event-Wise: {loose_events_result.count(1)}/{len(loose_events_result)}')
    
Metric("").calculate_point_wise(result_path="mydata_test_result")