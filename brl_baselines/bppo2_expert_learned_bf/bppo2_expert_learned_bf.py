import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from gym.spaces import Box
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from brl_baselines.bppo2_expert_learned_bf.runner import Runner
import torch
import multiprocessing
from brl_gym.estimators.learnable_bf import BayesFilterDataset
import brl_gym.estimators.learnable_bf.pt_util as pt_util
import brl_gym.estimators.learnable_bf.util as estimator_util
import torch.optim as optim

def constfn(val):
    def f(_):
        return val
    return f

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def learn(*, network, env, total_timesteps, expert, estimator_model,
            belief_dim, batch_size=96, sequence_length=32,
            estimator_epoch=5, estimator_lr=0.001, estimator_weight_decay=0.005,
            residual_weight=0.1,
            eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, update_starts_from=1,
            **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)
    print("bppo2_expert residual_weight", residual_weight)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)
    device = 'cuda'

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    ob_space = Box(np.concatenate([env.observation_space.low, env.action_space.low]), np.concatenate([env.observation_space.high, env.action_space.high]), dtype=np.float32)

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from brl_baselines.bppo2_expert.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, expert=expert, residual_weight=residual_weight)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam, expert=expert, residual_weight=residual_weight)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.time()

    train_loader = None
    data_train = None
    use_cuda = torch.cuda.is_available()
    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)
    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}

    optimizer = optim.Adam(estimator_model.parameters(),
        lr=estimator_lr, weight_decay=estimator_weight_decay)
    # estimator_model.load_last_model('estimator_xx_checkpoints_mse')
    # env.set_bayes_filter('estimator_xx_checkpoints_mse')
    estimator_log_path = 'estimator_logs_mse_brpo/log.pkl'
    PRINT_INTERVAL = 10

    estimator_train_losses, estimator_test_losses, estimator_test_accuracies = pt_util.read_log(estimator_log_path, ([], [], []))

    # env.set_bayes_filter('estimator_xx_checkpoints_mse')

    nupdates = total_timesteps//nbatch
    for update in range(int(update_starts_from), nupdates+1):
        print("update , ", update)
        assert nbatch % nminibatches == 0, "nbatch {} nminibatches {}".format(nbatch, nminibatches)
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, eplabels, epinfos = runner.run() #pylint: disable=E0632

        # train estimator_model
        # get input w/o belief
        actions = np.concatenate([np.zeros((1, actions.shape[1], actions.shape[2])), actions])[:-1, :, :]
        dones = masks.astype(np.int)
        dones = dones[:,:,None]
        data = np.concatenate([actions, obs[:, :, :-(actions.shape[2] + belief_dim)], dones], axis=2)
        data = data.transpose(1,0,2)
        label = eplabels.transpose()

        train_data = data
        train_label = label
        test_data = data
        test_label = label

        obs, returns, masks, actions, values, neglogpacs = map(sf01, (obs, returns, masks, actions, values, neglogpacs))

        # if data_train is None:
        data_train = BayesFilterDataset(train_data, train_label, belief_dim, sequence_length,
            batch_size=batch_size)
        data_test = BayesFilterDataset(test_data, test_label, belief_dim, sequence_length,
            batch_size=batch_size)
        # else:
        #     data_train.add_item(train_data, train_label)
        #     data_test.add_item(test_data, test_label)

        train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False, **kwargs)
        test_loader  = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, **kwargs)

        try:
            for epoch in range(0, estimator_epoch + 1):
                estimatorlr = estimator_lr #* np.power(0.25, (int(epoch / 6)))
                estimator_train_loss = estimator_util.train(
                    estimator_model, device, optimizer, train_loader, estimatorlr, epoch, PRINT_INTERVAL)
                estimator_test_loss, estimator_test_accuracy = estimator_util.test(
                    estimator_model, device, test_loader)
                estimator_train_losses.append((epoch, estimator_train_loss))
                estimator_test_losses.append((epoch, estimator_test_loss))
                estimator_test_accuracies.append((epoch, estimator_test_accuracy))
                pt_util.write_log(estimator_log_path, (estimator_train_losses, estimator_test_losses, estimator_test_accuracies))
                estimator_model.save_best_model(estimator_test_accuracy, 'estimator_checkpoints_mse_brpo/%03d.pt' % epoch)
            pass
        except KeyboardInterrupt as ke:
            print('Interrupted')
        except:
            import traceback
            traceback.print_exc()
        finally:
            # print('Saving final model')
            # estimator_model.save_model('estimator_checkpoints/%03d.pt' % epoch)
            """
            ep, val = zip(*estimator_train_losses)
            pt_util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
            """
            # ep, val = zip(*estimator_test_losses)
            #pt_util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
            # print("Test loss ", val[-1])
            """
            ep, val = zip(*estimator_test_accuracies)
            pt_util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')

            # Plot perplexity
            ep, val = zip(*estimator_train_losses)
            val = np.exp(val)
            pt_util.plot(ep, val, 'Train perplexity', 'Epoch', 'Error')
            """
            ep, val = zip(*estimator_test_losses)
            val = np.exp(val)
            # pt_util.plot(ep, val, 'Test perplexity', 'Epoch', 'Error')
            print("Final test perplexity was ", val[-1])

            # Update all envs' estimator
            env.set_bayes_filter('estimator_checkpoints_mse_brpo')
            # raise NotImplementedError

        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
        print('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
        print('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



