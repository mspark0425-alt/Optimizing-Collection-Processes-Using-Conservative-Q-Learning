import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

def generate_filename(base_name, args, extension="csv"):
    filename = f"{base_name}_lr{args.lr}_hs{args.hiddensize}_g{args.gamma}_a{args.alpha}_rc{args.rewardco}_se{args.seed}.{extension}"
    return os.path.join("results", filename)

# Find the next state and action
def mdp_process(state_batch, action_batch, reward_batch, device):
    tone_list = np.array([3, 6.5, 8.5, 10.5, 12, 13.25, 14.5, 15.5, 16, 16.25, 19, 20.5])
    ## Find the next state
    state_batch_df = pd.DataFrame(np.array(state_batch.cpu().clone()))
    if reward_batch == 'scenario':
        indices = [0]
    else:
        indices = torch.where((reward_batch[:,1] == 0))[0].tolist()
    
    # DLQ_Days
    DDA_condition_1 = (state_batch_df.loc[indices, 1] >= 1) & (state_batch_df.loc[indices, 1] <= 30)
    DDA_indices_1 = DDA_condition_1[DDA_condition_1 == True].index.values
    DDA_condition_2 = (state_batch_df.loc[indices, 1] >= 31) & (state_batch_df.loc[indices, 1] <= 60)
    DDA_indices_2 = DDA_condition_2[DDA_condition_2 == True].index.values
    DDA_condition_3 = (state_batch_df.loc[indices, 1] >= 61) & (state_batch_df.loc[indices, 1] <= 90)
    DDA_indices_3 = DDA_condition_3[DDA_condition_3 == True].index.values

    state_batch_df.loc[DDA_indices_1, 1] += np.random.randint(3, 6, size=DDA_condition_1.sum())
    state_batch_df.loc[DDA_indices_2, 1] += 1
    state_batch_df.loc[DDA_indices_2 & (np.random.rand(DDA_condition_2.sum()) < 0.5), 1] += 1
    state_batch_df.loc[DDA_indices_3, 1] += 1

    # Tone
    next_action_batch = action_batch.clone()
    next_indices_1 = np.random.rand(len(DDA_indices_1)) < 0.5
    next_action_batch[DDA_indices_1[next_indices_1]] = torch.cuda.FloatTensor(tone_list[np.clip(np.searchsorted(tone_list, action_batch[DDA_indices_1[next_indices_1]].cpu(), side='left'),None,11)])

    next_indices_2 = np.random.rand(len(DDA_indices_2)) < 0.25
    next_action_batch[DDA_indices_2[next_indices_2]] = torch.cuda.FloatTensor(tone_list[np.clip(np.searchsorted(tone_list, action_batch[DDA_indices_2[next_indices_2]].cpu(), side='left'),None,11)])

    next_indices_3 = np.random.rand(len(DDA_indices_3)) < 0.25
    next_action_batch[DDA_indices_3[next_indices_3]] = torch.cuda.FloatTensor(tone_list[np.clip(np.searchsorted(tone_list, action_batch[DDA_indices_3[next_indices_3]].cpu(), side='right'),None,11)])
    
    # Successful_Number_of_Actions_in_ever
    sna_condition = (next_action_batch == 16) | (next_action_batch == 19) | (next_action_batch == 20.5)
    sna_size = torch.nonzero(sna_condition.squeeze(-1)).squeeze(-1).cpu().numel()
    sna_indices = torch.nonzero(sna_condition.squeeze(-1)).squeeze(-1).cpu()[torch.rand(sna_size) < 0.45]

    state_batch_df.loc[:, 2] += 1
    if sna_indices.numel() > 0:
        state_batch_df.loc[sna_indices, 2] -= 1

    # Base_PD
    Base_PD_indices = DDA_indices_1[(state_batch_df.loc[DDA_indices_1, 1]> np.random.uniform(15, 25, size=len(DDA_indices_1)))].tolist() \
                        + DDA_indices_2[(state_batch_df.loc[DDA_indices_2, 1]> np.random.uniform(45, 55, size=len(DDA_indices_2)))].tolist() \
                        + DDA_indices_3[(state_batch_df.loc[DDA_indices_3, 1]> np.random.uniform(75, 85, size=len(DDA_indices_3)))].tolist()
    state_batch_df.loc[Base_PD_indices,0] = 1-(1-state_batch_df.loc[Base_PD_indices,0]) * np.exp(-np.log((1-state_batch_df.loc[Base_PD_indices,0])/0.03)*state_batch_df.loc[Base_PD_indices,1]/91)
    state_batch_df.loc[:, 0] = state_batch_df.loc[:, 0].clip(upper=0.99999)

    # Unsuccessful_Action_Ratio
    if sna_indices.numel() > 0:
        UAR_indices = torch.nonzero(sna_condition.squeeze(-1)).squeeze(-1).cpu()
        state_batch_df.loc[UAR_indices,7] = state_batch_df.loc[UAR_indices,7].values + torch.rand(torch.nonzero(sna_condition.squeeze(-1)).squeeze(-1).cpu().shape[0]).numpy()*0.3
        state_batch_df.loc[:, 7] = state_batch_df.loc[:, 7].clip(upper=0.99999)

    # Num_Dlq_Prod
    NDP_prob_mask = np.random.rand(len(indices)) < 0.05
    state_batch_df.loc[indices,4] += NDP_prob_mask.astype(int)

    # Delinquent_amount
    DA_prob_mask = (state_batch_df.loc[indices, 1] <= 60) & (state_batch_df.loc[indices, 1] > 30) & NDP_prob_mask
    DA_indices = np.where(DA_prob_mask)[0]
    state_batch_df.loc[state_batch_df.loc[DA_indices, 10][(state_batch_df.loc[DA_indices, 10] == 1)].index,5] *= 2
    state_batch_df.loc[state_batch_df.loc[DA_indices, 10][(state_batch_df.loc[DA_indices, 10] != 1)].index,5] *= (torch.rand(len(state_batch_df.loc[DA_indices, 10][(state_batch_df.loc[DA_indices, 10] != 1)].index)) * 0.5).numpy() + 1.5

    DA_prob_mask_2 = (state_batch_df.loc[indices, 1] > 60) & NDP_prob_mask
    DA_indices_2 = np.where(DA_prob_mask_2)[0]
    state_batch_df.loc[state_batch_df.loc[DA_indices_2, 12][(state_batch_df.loc[DA_indices_2, 12] == 1)].index,5] *= 1.5
    state_batch_df.loc[state_batch_df.loc[DA_indices_2, 12][(state_batch_df.loc[DA_indices_2, 12] != 1)].index,5] *= (torch.rand(len(state_batch_df.loc[DA_indices_2, 12][(state_batch_df.loc[DA_indices_2, 12] != 1)].index)) * 0.6).numpy() + 1.2

    # Tone_ever
    state_batch_df.loc[:, 11] = state_batch_df.loc[:, 11].values + next_action_batch.squeeze().cpu().numpy()

    next_state_batch = torch.tensor(state_batch_df.values).float().to(device)


    return next_state_batch, next_action_batch


    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def min_max_scaler(data, min_val='min', max_val='max'):
    if type(min_val)==str:
        min_val = data.min(axis=0)
        max_val = data.max(axis=0)
    return (data - min_val) / (max_val - min_val)

def inverse_min_max_scaler(data, min_val, max_val):
    return data * (max_val - min_val) + min_val


# Target Q-value computation function
def compute_target_q_values(reward_batch, next_state_batch, next_action_batch, gamma, target_network):
    with torch.no_grad():
        terminal_mask = (reward_batch[:,1] == 1)
        terminal_mask = terminal_mask.squeeze()
        non_terminal_mask = ~terminal_mask
        next_q_values = torch.zeros_like(reward_batch[:,0].unsqueeze(dim=1))
        next_q_values[non_terminal_mask] = target_network(next_state_batch[non_terminal_mask], next_action_batch[non_terminal_mask])
        target_q_values = reward_batch[:,0].unsqueeze(dim=1) + gamma * next_q_values
    return target_q_values

# CQL loss function
def cql_loss(predicted_q_values, target_q_values, conservative_loss_weight):
    mse_loss = F.mse_loss(predicted_q_values, target_q_values)
    conservative_loss = conservative_loss_weight * torch.logsumexp(predicted_q_values, dim=0)
    return mse_loss + conservative_loss

def choose_tone(action):
    if action == 'SMS1':
        tone = 3
    elif action == 'IVN1':
        tone = 6.5
    elif action == 'SMS2':
        tone = 8.5
    elif action == 'IVN2':
        tone = 10.5
    elif action == 'SMS3':
        tone = 12
    elif action == 'IVN3':
        tone = 13.25
    elif action == 'SMS4':
        tone = 14.5
    elif action == 'SMS5':
        tone = 15.5
    elif action == 'CC1':
        tone = 16
    elif action == 'SMS6':
        tone = 16.25
    elif action == 'CC2':
        tone = 19
    else:
        tone = 20.5
    return torch.tensor([[tone]], dtype=torch.float32)
    
def choose_action(recommended_action):
    if recommended_action.item() <= 3:
        action = 'SMS1'
    elif 3 < recommended_action.item() <= 6.5:
        action = 'IVN1'
    elif 6.5 < recommended_action.item() <= 8.5:
        action = 'SMS2'
    elif 8.5 < recommended_action.item() <= 10.5:
        action = 'IVN2'
    elif 10.5 < recommended_action.item() <= 12:
        action = 'SMS3'
    elif 12 < recommended_action.item() <= 13.25:
        action = 'IVN3'
    elif 13.25 < recommended_action.item() <= 14.5:
        action = 'SMS4'
    elif 14.5 < recommended_action.item() <= 15.5:
        action = 'SMS5'
    elif 15.5 < recommended_action.item() <= 16:
        action = 'CC1'
    elif 16 < recommended_action.item() <= 16.25:
        action = 'SMS6'
    elif 16.25 < recommended_action.item() <= 19:
        action = 'CC2'
    elif 19 < recommended_action.item() <= 20.5:
        action = 'CC3'
    return action

def assign_risk_segment(state):
    state = state.squeeze()
    if 1 <= state[1] <= 30:
        if 0 <= state[0] <= 0.136564:
            base = "Very Low"
        elif 0.136564 <= state[0] <= 0.275748:
            base = "Low"
        elif 0.275748 <= state[0] <= 0.387851:
            base = "Mid"
        elif 0.387851 <= state[0] <= 0.624628:
            base = "High"
        else:
            base = "Very High"

    elif 31 <= state[1] <= 60:
        if 0 <= state[0] <= 0.669768:
            base = "Mid"
        elif 0.669768 <= state[0] <= 0.890310:
            base = "High"
        else:
            base = "Very High"
    else:
        if 0 <= state[0] <= 0.9161701380:
            base = "High"
        else:
            base = "Very High"

    if (
        state[4] > 1
        or state[7] > 0.8
        or state[8] == 0
        or state[5] > 7130
    ):
        if base == "Very Low":
            base = "Low"
        elif base == "Low":
            base = "Mid"
        elif base == "Mid":
            base = "High"
        elif base == "High":
            base = "Very High"
    if base == "Very Low":
        return 4
    elif base == "Low":
        return 3
    elif base == "Mid":
        return 2
    elif base == "High":
        return 1
    else:
        return 0

def generate_rule_based_scenario(states, profits, action_table, state_min_val, state_max_val, action_min_val, action_max_val, args, device):
    total_reward = 0
    init_total_reward = 0
    results = []

    for i in range(states.shape[0]):
        current_state = states[i, :].clone().unsqueeze(0)
        inverse_current_state = inverse_min_max_scaler(current_state, state_min_val, state_max_val)
        init_delinquent_amount = inverse_current_state.squeeze()[5]
        profit = profits[i]
        collected = False
        churned = False
        cumulative_cost = 0
        daily_actions = []
        
        for day in range(int(inverse_current_state[0, 1]), 91):
            segment = assign_risk_segment(inverse_current_state)
            action = action_table[day][segment]

            if action is not None:
                action_tone = choose_tone(action).to(device)
                inverse_current_state = mdp_process(inverse_current_state, action_tone, 'scenario', device)[0]
                current_state = min_max_scaler(inverse_current_state, state_min_val, state_max_val)

                
                action_tone = choose_tone(action).to(device)
                daily_actions.append(action_tone.item())
                cumulative_cost += get_action_costs([action_tone.item()])[0]

                collection_prob, churn_prob = calculate_collection_and_churn_probability(current_state, action_tone)

                if not collected and np.random.rand() < collection_prob:
                    collected = True
                    break
                if not churned and np.random.rand() < churn_prob:
                    churned = True

                

        delinquent_amount = inverse_current_state.squeeze()[5]
        final_reward = delinquent_amount * int(collected) - args.rewardco * profit * int(churned) - cumulative_cost
        init_reward = init_delinquent_amount * int(collected) - args.rewardco * profit * int(churned) - cumulative_cost
        total_reward += final_reward
        init_total_reward += init_reward

        results.append({
            'customer_id': i,
            'collected': collected,
            'churned': churned,
            'final_reward': final_reward.item(),
            'init_reward' : init_reward.item(),
            'num_actions': len(daily_actions),
            'actions': daily_actions
        })

    df_result = pd.DataFrame(results)
    output_path = os.path.join("results", args.output_path)
    df_result.to_csv(output_path, index=False)
    print(f"[RULE] Scenario saved to {output_path}")
    print(f"[RULE] Total reward: {total_reward:.2f}")
    print(f"[RULE] Init Total reward: {init_total_reward:.2f}")
    return total_reward, init_total_reward, df_result


def generate_cql_based_scenario(states, profits, best_q_network, state_min_val, state_max_val, action_min_val, action_max_val, args, device):
    tone_list = torch.tensor([3, 6.5, 8.5, 10.5, 12, 13.25, 14.5, 15.5, 16, 16.25, 19, 20.5])
    total_reward = 0
    init_total_reward = 0
    results = []

    for i in range(states.shape[0]):
        current_state = states[i, :].clone().unsqueeze(0)
        inverse_current_state = inverse_min_max_scaler(current_state, state_min_val, state_max_val)
        init_delinquent_amount = inverse_current_state.squeeze()[5]
        profit = profits[i]
        collected = False
        churned = False
        cumulative_cost = 0
        daily_actions = []

        for day in range(int(inverse_current_state[0,1]), 91):
            if day != inverse_current_state[0, 1]:
                continue
            
            
            action_values = tone_list.to(device)
            scaled_action_values = min_max_scaler(action_values, action_min_val, action_max_val)
            q_values = best_q_network(current_state.repeat(tone_list.numel(), 1), scaled_action_values.unsqueeze(1)).squeeze()

            selected_idx = torch.argmax(q_values)
            action = action_values[selected_idx].unsqueeze(0)

            inverse_current_state = mdp_process(inverse_current_state, torch.reshape(action, (1, 1)), 'scenario', device)[0]
            current_state = min_max_scaler(inverse_current_state, state_min_val, state_max_val)

            daily_actions.append(action.item())
            cumulative_cost += get_action_costs([action])[0]
            
            collection_prob, churn_prob = calculate_collection_and_churn_probability(inverse_current_state, action.item())


            if not collected and np.random.rand() < collection_prob:
                collected = True
                break
            if not churned and np.random.rand() < churn_prob:
                churned = True
        
        
        delinquent_amount = inverse_current_state.squeeze()[5]
        final_reward = delinquent_amount * int(collected) - args.rewardco * profit * int(churned) - cumulative_cost
        init_reward = init_delinquent_amount * int(collected) - args.rewardco * profit * int(churned) - cumulative_cost
        total_reward += final_reward
        init_total_reward += init_reward
        results.append({
            'customer_id': i,
            'collected': collected,
            'churned': churned,
            'final_reward': final_reward.item(),
            'init_reward' : init_reward.item(),
            'num_actions': len(daily_actions),
            'actions': daily_actions
        })

    df_result = pd.DataFrame(results)
    output_path = generate_filename(args.output_path, args, extension='csv')
    df_result.to_csv(output_path, index=False)
    print(f"[CQL] Scenario saved to {output_path}")
    print(f"[CQL] Total reward: {total_reward:.2f}")
    print(f"[CQL] Init Total reward: {init_total_reward:.2f}")
    return total_reward, init_total_reward, df_result



def get_action_costs(tone_values):
    tone_to_cost = {
        3: 0.01,
        6.5: 0.02,
        8.5: 0.01,
        10.5: 0.02,
        12: 0.01,
        13.25: 0.02,
        14.5: 0.01,
        15.5: 0.01,
        16: 3,
        16.25: 0.01,
        19: 3,
        20.5: 3
    }
    return np.array([tone_to_cost.get(t, 0) for t in tone_values], dtype=np.float32)

import numpy as np
import torch

def calculate_collection_and_churn_probability(state_row, action_tone):
    state_row = state_row.squeeze().cpu().numpy()
    if isinstance(action_tone, torch.Tensor):
        action_tone = action_tone.item() 

    ### ===== Collection Probability =====
    c = 0
    c += -0.0162 * state_row[1]         # DLQ_Days
    c += -0.0722 * action_tone          # current action tone
    c += 0.0023 * state_row[2]          # Successful_Number_of_Actions_in_ever

    # Exitin20_Last6M
    if state_row[6] <= 0.064:
        c += -0.2855
    elif state_row[6] <= 0.304:
        c += -0.1163
    elif state_row[6] <= 0.576:
        c += 0.069
    else:
        c += 0.1063

    # Unsuccessful_Action_Ratio
    if state_row[7] <= 0.022:
        c += 0.5298
    elif state_row[7] <= 0.176:
        c += -0.0792
    elif state_row[7] <= 0.264:
        c += -0.2645
    elif state_row[7] <= 0.5:
        c += -0.4127

    # Days_btw_delinquency
    if state_row[9] <= 14:
        c += -0.1826
    elif state_row[9] <= 91:
        c += -0.044

    # Num_Dlq_Prod
    if state_row[4] <= 2:
        c += 0.0258
    elif state_row[4] <= 5:
        c += -0.0865
    elif state_row[4] <= 10:
        c += -0.1658

    # Delinquent_amount
    if state_row[5] <= 332.78:
        c += 0.2785
    elif state_row[5] <= 652.78:
        c += -0.1693
    elif state_row[5] <= 29652.53:
        c += -0.3358

    c += -0.261  # Intercept

    collection_prob = 1 / (1 + np.exp(-c))

    ### ===== Churn Probability =====
    ch = 0
    ch += 0.0125 * state_row[1]         # DLQ_Days
    ch += 0.0121 * action_tone          # current action tone
    ch += -0.0001 * state_row[2]        # Successful_Number_of_Actions_in_ever
    ch += 0.1136 * state_row[8]         # Payment_ratio_last3months

    # Num_Prod
    if state_row[3] <= 1:
        ch += -0.2977
    elif state_row[3] <= 5:
        ch += -0.4441
    elif state_row[3] <= 11:
        ch += -0.6127

    # Days_btw_delinquency
    if state_row[9] <= 2:
        ch += -0.262
    elif state_row[9] <= 44:
        ch += -0.4306
    elif state_row[9] <= 91:
        ch += -0.662

    ch += -1.377  # Intercept
    churn_prob = 1 / (1 + np.exp(-ch))

    return collection_prob, churn_prob
