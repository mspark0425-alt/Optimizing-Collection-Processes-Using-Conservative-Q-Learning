import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from sklearn.model_selection import train_test_split
import warnings
import ast
warnings.filterwarnings('ignore')
from scipy import stats

from util import *
from Network import *
from action_table import *

############ Main function
def main(args):
    
    os.makedirs("results", exist_ok=True)
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)

    # Load data
    data = pd.read_excel(args.data_path)
    data = data.dropna(axis=1).reset_index(drop=True)
    data['CC'] = np.where(data['Delinquent_product'] == 'CC', 1, 0)
    data['GPL'] = np.where(data['Delinquent_product'] == 'GPL', 1, 0)

    # Set state, action data
    states = data[['Base_PD', 'DLQ_Days', 'Successful_Number_of_Actions_in_ever', 'Num_Prod', 'Num_Dlq_Prod',
                  'Delinquent_amount', 'Exitin20_Last6M', 'Unsuccessful_Action_Ratio',
                  'Payment_ratio_last3months','Days_btw_delinquency','GPL','Tone_ever', 'CC']].values.astype(np.float32)
    actions = data['Tone'].values.astype(np.float32)
    costs = get_action_costs(actions)
    # Make a reward column
    rewards = data['Delinquent_amount'].values.astype(np.float32) * data['Target'].values.astype(np.float32) - args.rewardco * data['profit'].values.astype(np.float32) * data['Target_Churn'].values.astype(np.float32) - costs

    rewards = np.vstack([rewards, data['Target'].values.astype(np.float32)])
    
    # Calculate the min and max values for states and actions
    state_min_val = torch.tensor(states.min(axis=0)).to(device)
    state_max_val = torch.tensor(states.max(axis=0)).to(device)

    action_min_val = torch.tensor(actions.min()).to(device)
    action_max_val = torch.tensor(actions.max()).to(device)

    # Normalize states and actions using the min-max scaler
    states = min_max_scaler(states)
    actions = min_max_scaler(actions)
    states = torch.tensor(states).to(device)
    actions = torch.tensor(actions).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards.T).to(device)

    # Split data by Customer ID
    unique_customers = data['Customer_ID_Before'].dropna().unique()
    train_customers, temp_customers = train_test_split(unique_customers, test_size=0.5, shuffle=True, random_state= args.seed)
    val_customers, test_customers = train_test_split(temp_customers, test_size=0.6, shuffle=True, random_state= args.seed)

    train_mask = data['Customer_ID_Before'].isin(train_customers).values
    val_mask = data['Customer_ID_Before'].isin(val_customers).values
    test_mask = data['Customer_ID_Before'].isin(test_customers).values

    train_states, val_states = states[train_mask], states[val_mask]
    train_actions, val_actions = actions[train_mask], actions[val_mask]
    train_rewards, val_rewards = rewards[train_mask], rewards[val_mask]
    
    test_data = data[test_mask].reset_index(drop=True)
    first_state_indices = test_data.groupby('Customer_ID_Before').head(1).index
    test_states = states[test_mask][first_state_indices]

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_states, train_actions, train_rewards)
    val_dataset = TensorDataset(val_states, val_actions, val_rewards)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

    # Initialize Q-network and target network
    state_dim = states.shape[1] - 2
    action_dim = 1
    hidden_dim = args.hiddensize
    gamma = args.gamma
    alpha = args.alpha
    q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    
    ### CQL ###
    if args.load_new:

        target_network.load_state_dict(q_network.state_dict())

        optimizer = optim.Adam(q_network.parameters(), lr=args.lr)

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0

        num_epochs = args.epochs
        target_update_interval = args.targetupdateinterval

        # Training loop
        for epoch in range(num_epochs):
            print(epoch)
            for state_batch, action_batch, reward_batch in train_dataloader:
                # Move batches to device
                state_batch = state_batch.to(device)
                action_batch = action_batch.to(device)
                reward_batch = reward_batch.to(device)
            
                # Inverse scaling before MDP process using global min and max values
                original_state_batch = inverse_min_max_scaler(state_batch, state_min_val, state_max_val)
                original_action_batch = inverse_min_max_scaler(action_batch, action_min_val, action_max_val)

                # MDP processing
                next_state_batch, next_action_batch = mdp_process(original_state_batch, original_action_batch, reward_batch, device)

                # Reapply scaling after MDP process using global min and max values
                next_state_batch = min_max_scaler(next_state_batch, state_min_val, state_max_val)
                next_action_batch = min_max_scaler(next_action_batch, action_min_val, action_max_val)

                # Continue with the training steps
                predicted_q_values = q_network(state_batch, action_batch)
                target_q_values = compute_target_q_values(reward_batch, next_state_batch, next_action_batch, gamma, target_network)
                loss = cql_loss(predicted_q_values, target_q_values, alpha)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation phase
            val_loss = 0.0
            with torch.no_grad():
                for val_state_batch, val_action_batch, val_reward_batch in val_dataloader:
                    # Move batches to device
                    val_state_batch = val_state_batch.to(device)
                    val_action_batch = val_action_batch.to(device)
                    val_reward_batch = val_reward_batch.to(device)

                    # Inverse scaling before MDP process using global min and max values
                    original_val_state_batch = inverse_min_max_scaler(val_state_batch, state_min_val, state_max_val)
                    original_val_action_batch = inverse_min_max_scaler(val_action_batch, action_min_val, action_max_val)

                    # MDP processing
                    next_val_state_batch, next_val_action_batch = mdp_process(original_val_state_batch, original_val_action_batch, val_reward_batch, device)

                    # Reapply scaling after MDP process using global min and max values
                    next_val_state_batch = min_max_scaler(next_val_state_batch, state_min_val, state_max_val)
                    next_val_action_batch = min_max_scaler(next_val_action_batch, action_min_val, action_max_val)

                    # Continue with the validation steps
                    val_q_values = q_network(val_state_batch, val_action_batch)
                    target_q_values = compute_target_q_values(val_reward_batch, next_val_state_batch, next_val_action_batch, gamma, target_network)
                    loss = cql_loss(val_q_values, target_q_values, alpha)
                    val_loss += loss.item()

            average_val_loss = val_loss / len(val_dataloader)

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                patience_counter = 0
                model_path = generate_filename("best_q_network", args, extension="pth")
                torch.save(q_network.state_dict(), model_path)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                print(epoch)
                break

            if epoch % target_update_interval == 0:
                target_network.load_state_dict(q_network.state_dict())

        print("Training finished.")




    print("Predicting optimal actions:")
    
    best_q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    model_path = generate_filename("best_q_network", args, extension="pth")
    best_q_network.load_state_dict(torch.load(model_path))
    best_q_network.eval()

    profits = data[test_mask].reset_index(drop=True)['profit'].values[first_state_indices]
    print("Generating scenario...")


    gpl_idx = (test_states[:, -3] == 1).nonzero(as_tuple=False).squeeze(1).cpu().numpy()
    cc_idx = (test_states[:, -1] == 1).nonzero(as_tuple=False).squeeze(1).cpu().numpy()
    od_idx = ((test_states[:, -3] == 0) & (test_states[:, -1] == 0)).nonzero(as_tuple=False).squeeze(1).cpu().numpy()
    print(f'GPL : {len(gpl_idx)}, CC : {len(cc_idx)}, OD :  {len(od_idx)}')



    cql_reward_list = []
    cql_init_reward_list = []

    rule_reward_list = []
    rule_init_reward_list = []
    cql_reward, cql_init_reward, cql_result = generate_cql_based_scenario(
        test_states,
        profits,
        best_q_network,
        state_min_val,
        state_max_val,
        action_min_val,
        action_max_val,
        args,
        device
    )

    rule_reward, rule_init_reward, rule_result = generate_rule_based_scenario(
        test_states,
        profits,
        action_table,
        state_min_val,
        state_max_val,
        action_min_val,
        action_max_val,
        args,
        device
    )



    cql_reward_list.append(cql_reward)
    cql_init_reward_list.append(cql_init_reward)

    rule_reward_list.append(rule_reward)
    rule_init_reward_list.append(rule_init_reward)
    cql_final = np.array(cql_result['final_reward'], dtype=float)

    rule_final = np.array(rule_result['final_reward'], dtype=float)
    cql_init = np.array(cql_result['init_reward'], dtype=float)

    rule_init = np.array(rule_result['init_reward'], dtype=float)
    t_stat_final, p_one_final = stats.ttest_rel(cql_final, rule_final, alternative='greater')
    t_stat_init, p_one_init = stats.ttest_rel(cql_init, rule_init, alternative='greater')


    improvement = 100 * (sum(cql_reward_list)/len(cql_reward_list) - sum(rule_reward_list)/len(rule_reward_list)) / abs(sum(rule_reward_list))
    init_improvement = 100 * (sum(cql_init_reward_list)/len(cql_init_reward_list) - sum(rule_init_reward_list)/len(rule_init_reward_list)) / abs(sum(rule_init_reward_list))

    gpl_improvement = 100*(cql_result['final_reward'][gpl_idx].sum()-rule_result['final_reward'][gpl_idx].sum())/rule_result['final_reward'][gpl_idx].sum()
    init_gpl_improvement = 100*(cql_result['init_reward'][gpl_idx].sum()-rule_result['init_reward'][gpl_idx].sum())/rule_result['init_reward'][gpl_idx].sum()
    p_one_final_gpl = stats.ttest_rel(cql_final[gpl_idx], rule_final[gpl_idx], alternative='greater')[1]
    p_one_init_gpl = stats.ttest_rel(cql_init[gpl_idx], rule_init[gpl_idx], alternative='greater')[1]


    cc_improvement = 100*(cql_result['final_reward'][cc_idx].sum()-rule_result['final_reward'][cc_idx].sum())/rule_result['final_reward'][cc_idx].sum()
    init_cc_improvement = 100*(cql_result['init_reward'][cc_idx].sum()-rule_result['init_reward'][cc_idx].sum())/rule_result['init_reward'][cc_idx].sum()
    p_one_final_cc = stats.ttest_rel(cql_final[cc_idx], rule_final[cc_idx], alternative='greater')[1]
    p_one_init_cc = stats.ttest_rel(cql_init[cc_idx], rule_init[cc_idx], alternative='greater')[1]

    od_improvement = 100*(cql_result['final_reward'][od_idx].sum()-rule_result['final_reward'][od_idx].sum())/rule_result['final_reward'][od_idx].sum()
    init_od_improvement = 100*(cql_result['init_reward'][od_idx].sum()-rule_result['init_reward'][od_idx].sum())/rule_result['init_reward'][od_idx].sum()
    p_one_final_od = stats.ttest_rel(cql_final[od_idx], rule_final[od_idx], alternative='greater')[1]
    p_one_init_od = stats.ttest_rel(cql_init[od_idx], rule_init[od_idx], alternative='greater')[1]

    print(f"CQL is {improvement:.4f}% better than Rule!(Final)")
    print(f"CQL is {init_improvement:.4f}% better than Rule!(Init)")
    print(f"t-statistic_final = {t_stat_final:.4f}, one-sided p-value_final = {p_one_final:.6f}")
    print(f"t-statistic_init = {t_stat_init:.4f}, one-sided p-value_init = {p_one_init:.6f}")


    # Save improvement and parameters to Excel
    improvement_data = {
        'seed': [args.seed],
        'rewardco': [args.rewardco],
        'batchsize': [args.batchsize],
        'lr': [args.lr],
        'hiddensize': [args.hiddensize],
        'patience': [args.patience],
        'gamma': [args.gamma],
        'alpha': [args.alpha],
        'epochs': [args.epochs],
        'targetupdateinterval': [args.targetupdateinterval],
        'top_k': [args.top_k],
        'data_path': [args.data_path],
        'output_path': [args.output_path],
        'improvement_percent_final': [float(improvement)],
        'improvement_percent_init': [float(init_improvement)],
        'p-value_final' : [float(p_one_final)],
        'p-value_init' : [float(p_one_init)],
        'cql_reward': [float(cql_reward)],
        'rule_reward': [float(rule_reward)],
        'cql_init_reward': [float(cql_init_reward)],
        'rule_init_reward': [float(rule_init_reward)],
        'gpl_improvement' : [float(gpl_improvement)],
        'init_gpl_improvement' : [float(init_gpl_improvement)],
        'p_one_final_gpl' : [float(p_one_final_gpl)],
        'p_one_init_gpl' : [float(p_one_init_gpl)],
        'cc_improvement' : [float(cc_improvement)],
        'init_cc_improvement' : [float(init_cc_improvement)],
        'p_one_final_cc' : [float(p_one_final_cc)],
        'p_one_init_cc' : [float(p_one_init_cc)],
        'od_improvement' : [float(od_improvement)],
        'init_od_improvement' : [float(init_od_improvement)],
        'p_one_final_od' : [float(p_one_final_od)],
        'p_one_init_od' : [float(p_one_init_od)],
    }


    improvement_df = pd.DataFrame(improvement_data)
    improvement_path = "results/improvement.csv"

    improvement_df.to_csv(improvement_path, mode='a', header=not os.path.exists(improvement_path), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=8, help="Random seed for reproducibility")
    parser.add_argument('--load_new', type=ast.literal_eval, default=True, help="Train new modl")
    parser.add_argument('--rewardco', type=int, default=5, help="Reward coefficient")
    parser.add_argument('--batchsize', type=int, default=1024, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for the optimizer")
    parser.add_argument('--hiddensize', type=int, default=64, help="Size of hidden layers in Q-network")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--gamma', type=float, default=0.9, help="Discount factor for future rewards")
    parser.add_argument('--alpha', type=float, default=0.5, help="Weight for conservative loss in CQL")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--targetupdateinterval', type=int, default=5, help="Target network update interval")
    parser.add_argument('--top_k', type=int, default=3, help="Top-k actions to choose from when type=random")
    parser.add_argument('--data_path', type=str, default='New_DS_60k.xlsx', help="Path to the input data file")
    parser.add_argument('--number', type=int, default=100, help="Number of customers")
    parser.add_argument('--output_path', type=str, default='scenario_output.csv', help="Path to the output data file")
    args = parser.parse_args()

    main(args)
