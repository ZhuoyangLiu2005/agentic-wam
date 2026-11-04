"""RoboDojo episode loop for AgenticWAM: one observation per chunk, then the chunk's rows (deploy.yml replan_steps)."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        for action in model_client.call(func_name="get_action"):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    # deploy.yml sets eval_batch: false, so the sim runs one env; kept for the XPolicyLab contract.
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        for action_idx in range(len(actions[0])):
            TASK_ENV.take_action_batch([a[action_idx] for a in actions], env_idx_list)
            if TASK_ENV.is_episode_end():
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            keep = [i for i, e in enumerate(env_idx_list) if e in running]
            actions, env_idx_list = [actions[i] for i in keep], [env_idx_list[i] for i in keep]
