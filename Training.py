# defalut setting
last_bit_rate = 0
bit_rate = 0
target_buffer = 0
latency_limit = 4

# QOE setting
reward_frame = 0
reward_all = 0
SMOOTH_PENALTY= 0.02
REBUF_PENALTY = 1.85
LANTENCY_PENALTY = 0.005
SKIP_PENALTY = 0.5

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
#         _, reward, done, _ = env.step(action.item())
        time,time_interval, send_data_size, chunk_len, rebuf, buffer_size, play_time_len,end_delay,\
                cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag, \
                buffer_flag, cdn_flag, skip_flag,end_of_video = net_env.get_video_frame(bit_rate,target_buffer, latency_limit)
        if not cdn_flag:
            reward = frame_time_len * float(BIT_RATE[bit_rate]) / 1000  - REBUF_PENALTY * rebuf - LANTENCY_PENALTY  * end_delay - SKIP_PENALTY * skip_frame_time_len 
        else:
            reward = -(REBUF_PENALTY * rebuf)
        if decision_flag or end_of_video:
            # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
            reward += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            # last_bit_rate
            last_bit_rate = bit_rate
            # Select and perform an action
            action = select_action(time,time_interval, send_data_size, chunk_len, rebuf, buffer_size, play_time_len,end_delay,\
                cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag, \
                buffer_flag, cdn_flag, skip_flag,end_of_video)
            
        reward = torch.tensor([reward], device=device)
        # Store the transition in memory
        memory.push(state, action, reward)
        
        
        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
