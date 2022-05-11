import numpy as np
import pandas as pd
import torch
import time
import random
import gc
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
tqdm.pandas()

from util import *
from matplotlib.lines import Line2D


def plot_fake_mels(gen, fixed_noise,fixed_real,fixed_vector,labels,
                   epoch_ii,plot_save_after_i_iterations,
                   dict_file,file_to_store,
                   starting_i = 0):
    """
        Plot 6 generated log-mel spectrograms from a fixed noise Tensor in comparison to 6 fixed real examples

        :param gen: torch model (generator)
            generator model
        :param fixed_noise: 3D torch.Tensor (batch, 1, z_dim)
            Tensor containing a random noise vector for each example to generate from batch
        :param fixed_real: 3D torch.Tensor (batch, time, mel_channel)
            Tensor containing a batch of real examples
        :param fixed_vector: 2D torch.Tensor (batch, sem_vec)
            Tensor containing the corresponding semantic vector for each sample in fixed_real
        :param epoch_ii: int
            current epoch in training
        :param plot_save_after_i_iterations: int
            number of epochs to train model before saving generated examples
        :param fixed_files: pd.Series
            corresponding file to each real example
        :param dict_file: str
            name of the dictionary to save the plot in
        :param file_to_store: str
            file name for saving the plot
        :param starting_i: int
            index in batch at which we start to take 6 succesive examples (e.g. ensure to take 3 recorded and 3 synthesized examples)
        :return: None
        """

    with torch.no_grad():
        fake = gen(fixed_noise, len(fixed_real[0]), fixed_vector)
        fig, axes = plt.subplots(nrows=4, ncols=3, facecolor="white", figsize=(15, 10), sharey=True)
        i = starting_i  # e.g. ensure 3 recorded 3 synthesized examples
        for row in range(2):
            for col in range(3):
                # if i < 8:
                ax1 = axes[row * 2, col]
                ax2 = axes[row * 2 + 1, col]

                real_img = fixed_real[i].detach().cpu().T
                fake_img = fake[i].detach().cpu().T
                ax1.imshow(real_img, origin='lower')
                ax2.imshow(fake_img, origin='lower')
                
                ax1.set_title(labels.iloc[i], fontsize=18, pad=5)

                ax1.set_xticks([])
                ax2.set_xticks([])
                ax1.yaxis.tick_right()
                ax2.yaxis.tick_right()

                if col == 2:
                    ax1.yaxis.set_label_position("right")
                    ax1.set_ylabel('Mel Channel', fontsize=15, rotation=270, labelpad=20)
                    ax2.yaxis.set_label_position("right")
                    ax2.set_ylabel('Mel Channel', fontsize=15, rotation=270, labelpad=20)

                else:
                    ax1.set_yticks([])
                    ax2.set_yticks([])

                i += 1
        #fig.subplots_adjust(hspace=-0.03)
        axes[0, 0].text(-0.1, 0.5, "Real", fontsize=18, va="center", rotation=90,
                        transform=axes[0, 0].transAxes)
        axes[1, 0].text(-0.1, 0.5, "Fake", fontsize=18, va="center", rotation=90,
                        transform=axes[1, 0].transAxes)
        axes[2, 0].text(-0.1, 0.5, "Real", fontsize=18, va="center", rotation=90,
                        transform=axes[2, 0].transAxes)
        axes[3, 0].text(-0.1, 0.5, "Fake", fontsize=18, va="center", rotation=90,
                        transform=axes[3, 0].transAxes)
        fig.subplots_adjust(hspace=0.2, wspace=-0.0
                            )


        if (epoch_ii + 1) % plot_save_after_i_iterations == 0:
            plt.savefig(f"{dict_file + file_to_store}_{epoch_ii + 1}.png")
        plt.show()

def plot_fake_cps(gen, fixed_noise,fixed_real, fixed_vector, labels,
                   epoch_ii,plot_save_after_i_iterations,
                   dict_file,file_to_store, colors, n_cps = 5):
    """
            Plot the first 5 cp-trajectories from 9 different words in comparison with the same cps generated from a fixed noise Tensor

            :param gen: torch model (generator)
                generator model
            :param fixed_noise: 3D torch.Tensor (batch, 1, z_dim)
                Tensor containing a random noise vector for each example to generate from batch
            :param fixed_real: 3D torch.Tensor (batch, time, cp)
                Tensor containing a batch of real examples
            :param fixed_vector: 2D torch.Tensor (batch, sem_vec)
                Tensor containing the corresponding semantic vector for each sample in fixed_real
            :param epoch_ii: int
                current epoch in training
            :param plot_save_after_i_iterations: int
                number of epochs to train model before saving generated examples
            :param fixed_files: pd.Series
                corresponding file to each real example
            :param dict_file: str
                name of the dictionary to save the plot in
            :param file_to_store: str
                file name for saving the plot
            :param colors: list
                list of colors for different cps
            :param n_cps: int (default = 5)
                number of cps to plot
            :return: None
            """
    assert len(colors) <= n_cps, "Not enough colors provide for plotting distinct cp-trajectories: %d provided %d needed!" % (len(colors), n_cps)
    with torch.no_grad():
        fake = gen(fixed_noise,len(fixed_real[0]),fixed_vector)
        fig, axes = plt.subplots(nrows = 3, ncols=3, facecolor = "white", figsize = (15,10))
        i = 0
        for row in range(3):
            for col in range(3):
                ax = axes[row,col]
                for c in range(n_cps):
                    if i < len(fixed_real):
                        ax.plot(fake[i][:,c].detach().cpu(),color = colors[c])
                        ax.plot(fixed_real[i][:,c].detach().cpu(), color = colors[c], linestyle = "dotted")
                    if row < 2:
                        ax.set_xticks([])
                    if col > 0:
                        ax.set_yticks([])
                if i < len(fixed_real):
                    ax.set_title(labels.iloc[i],fontsize=18, pad=7)
                ax.set_ylim((-1.1,1.1))
                i+=1
        fig.subplots_adjust(hspace = 0.2, wspace = 0.1)

        if (epoch_ii + 1) % plot_save_after_i_iterations == 0:
            plt.savefig(f"{dict_file + file_to_store}_{epoch_ii + 1}.png")
        plt.show()


def plot_cp_mean_comparison(generator, fixed_noise, fixed_reals, fixed_lens,
                            fixed_vectors, label,
                            epoch_ii, plot_save_after_i_iterations, dict_file, file_to_store="test",
                            device="cpu"):
    padded_reals = pad_batch_online(fixed_lens, fixed_reals, device)
    mean_cp = torch.mean(padded_reals, axis=0)

    generator.eval()
    with torch.no_grad():
        generated_cps = []
        for i, l in enumerate(fixed_lens):
            noise_i = fixed_noise[i]
            noise_i = torch.unsqueeze(noise_i, 0)
            with torch.no_grad():
                generated_cps += [generator(noise_i, l, fixed_vectors[i:i + 1]).detach().cpu()[0]]

    # generated_cps = pad_batch_online(fixed_lens, pd.Series(generated_cps), device = device)
    colors = ["C%d" % i for i in range(5)]
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor="white", figsize=(15, 10))
    for c in range(5):
        for cp in generated_cps[:5]:
            ax.plot(cp[:, c], color=colors[c], lw=3, linestyle="solid")
        for cp in fixed_reals[:5]:
            ax.plot(cp[:, c], color=colors[c], linestyle="solid", lw=3, alpha=0.3)
        ax.plot(mean_cp[:, c].detach().cpu(), color="black", linestyle="dotted", lw=4)

    legend_elements = [Line2D([0], [0], color='black', ls="dotted", lw=3, label='Mean CPs'),  #
                       Line2D([0], [0], color='black', ls="solid", lw=3, alpha=0.3, label='Segment CPs'),
                       Line2D([0], [0], color='black', ls="solid", lw=3, label='Generated CPs')]

    ax.set_ylim((-1.1, 1.1))
    plt.legend(handles=legend_elements, fontsize=15, bbox_to_anchor=(1.0, 1), frameon=False)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel('Normalized Position', fontsize=20, labelpad=20)
    ax.set_xlabel('Timestep (2.5ms)', fontsize=20, labelpad=20)
    plt.title("CPs for label: '%s'" % label, fontsize=25, pad=20)

    if (epoch_ii + 1) % plot_save_after_i_iterations == 0:
        plt.savefig(f"{dict_file + file_to_store}_{epoch_ii + 1}.pdf", bbox_inches='tight')


class Training:
    """
        Create Training Instance
        :param gen: torch model
            generator model to train
        :param critic: torch model
            critic model to train
        :param seed: int or str
            int to set random.seed in order to be reproducible
        :param torch_seed: int or str
            int to set torch.manual_seed in order to be reproducible
        :param target_name: str (one of ["mel", "cp"]) nedded to call correct plotting function
        :param inps: pd.Series
            series containing the inputs of the training set
        :param vectors: pd.Series
            series containing the corresponding semantic vectors
        :parram z_dim: int
            number of noise dimensions
        :param batch_size: int
            batch size
        :param res_train: pd.DataFrame
            pd.DataFrame for logging epoch results on training set

        :param opt_gen: torch.optim.Optimizer
            torch optimizer for updating weights of the generator model
        :param opt_critic: torch.optim.Optimizer
            torch optimizer for updating weights of the critic model

        :param use_same_size_batching: bool
            specify whether to batch inps with similar length during epoch creating in order to avoid long padding

        :param files: pd.Series
            corresponding file to each real example

        """
    def __init__(self, gen, critic, seed, torch_seed,tgt_name, inps,labels, vectors,z_dim,
                 batch_size, res_train , opt_gen, opt_critic,use_same_size_batching = True,
                 inps_valid=None, labels_valid=None, vectors_valid=None,
                 pre_res_train = None, pre_res_valid=None, pre_batch_size = None, pre_opt_gen=None, pre_criterion=None, plot_mean_comparison_to = None ):

        self.seed = seed
        self.torch_seed = torch_seed
        self.gen = gen
        self.critic = critic
        self.opt_gen = opt_gen
        self.opt_critic = opt_critic
        self.pre_opt_gen = pre_opt_gen
        self.pre_criterion = pre_criterion

        self.device = next(gen.parameters()).device
        self.use_same_size_batching = use_same_size_batching

        assert tgt_name in ["mel", "cp"], "Please provide a valid tgt name (mel or cp)!"
        self.tgt_name = tgt_name
        self.inps = inps
        self.inps_valid = inps_valid
        self.labels = labels
        self.labels_valid = labels_valid
        
        self.vectors = torch.Tensor(np.array(list(vectors))).double().to(self.device)
        if vectors_valid is not None:
            self.vectors_valid = torch.Tensor(np.array(list(vectors_valid))).double().to(self.device)
        # get lengths of inputs and outputs
        self.lens = torch.Tensor(np.array(inps.apply(lambda x: len(x)), dtype=int)).to(self.device)
        if inps_valid is not None:
            self.lens_valid = torch.Tensor(np.array(inps_valid.apply(lambda x: len(x)), dtype=int)).to(self.device)
        
        self.z_dim = z_dim
        #if noise_mean is None:
        #    self.noise_mean = np.zeros(z_dim)
        #if noise_std is None:
        #    self.noise_std = np.eye(z_dim)
        
        self.batch_size = batch_size
        self.pre_batch_size = pre_batch_size

        self.res_train = res_train
        self.res_train_ix = len(res_train)
        self.pre_res_train = pre_res_train
        self.pre_res_train_ix = len(pre_res_train)
        self.pre_res_valid = pre_res_valid
        if pre_res_valid is not None:
            self.pre_res_valid_ix = len(pre_res_valid)
            
        self.plot_mean_comparison_to = plot_mean_comparison_to
        
        #self.decomposition_matrix, self.decomposition = get_decomposition_matrix(self.noise_std)

        # is using same size batching we create a dictionary containing all unique lengths and the indices of each sample with a this length
        if use_same_size_batching:
            self.train_length_dict = {}
            lengths, counts = np.unique(self.lens.cpu(), return_counts=True)
            self.sorted_length_keys = np.sort(lengths)

            for length in self.sorted_length_keys:
                self.train_length_dict[length] = np.where(self.lens.cpu() == length)[0]

        # set seed and create fixed random noise vectors and a fixed batch of real examples to monitor training
        random.seed(self.seed)
        torch.manual_seed(self.torch_seed)

        if self.plot_mean_comparison_to is None:
            self.fixed_batch = self.create_epoch_batches(len(self.inps), batch_size=self.batch_size, same_size_batching=True)[0]
            self.fixed_vectors = self.vectors[self.fixed_batch]
            self.fixed_lens = self.lens[self.fixed_batch]
            self.fixed_noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in self.fixed_lens])
            #self.fixed_noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(self.fixed_lens)])
            self.fixed_noise = pad_batch_online(self.fixed_lens, self.fixed_noise , self.device)
            self.fixed_reals = self.inps.iloc[self.fixed_batch]
            self.fixed_reals = pad_batch_online(self.fixed_lens, self.fixed_reals, self.device)
            self.fixed_labels = self.labels.iloc[self.fixed_batch]
        else:
            self.fixed_vectors =self.vectors[np.asarray(self.labels == self.plot_mean_comparison_to)]
            self.fixed_lens = self.lens[np.asarray(self.labels == self.plot_mean_comparison_to)]
            self.fixed_noise = [torch.randn(int(l.item()), self.z_dim,device=self.device) for l in self.fixed_lens]
            #self.fixed_noise = [np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _, l in enumerate(self.fixed_lens)]
            self.fixed_reals = self.inps.iloc[np.asarray(labels == self.plot_mean_comparison_to)]
	    	

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def gradient_penalty(self, critic, lens, vectors, real, fake, device="cpu"):
        """
        Gradient Penalty to enforce the Lipschitz constraint in order for the critic to be able to approximate a valid
        1-Lipschitz function. This is needed in order to use the Kantorovich-Rubinstein duality for simplifying the
        calculation of the wasserstein distance

        :param critic: torch model
            critic model
        :param lens: 2D torch.Tensor
            Tensor containing the real unpadded length of each sample in batch (batch, length)
        :param vectors: 2D torch.Tensor
            Tensor containing the corresponding semantic vectors
        :param real: 3D torch.Tensor (batch, time, cps)
            Tensor containing real images
        :param fake: 3D torch.Tensor (batch, time, cps)
            Tensor containing generated images
        :param device: str
            device to run calculation on
        :return:
        """
        batch_size, length, c = real.shape
        alpha = torch.rand((batch_size, 1, 1)).repeat(1, length, c).to(device)

        interpolated_input = real * alpha + fake * (1 - alpha)

        # Calculate critic scores
        mixed_scores = critic(interpolated_input, lens, vectors)
        
        #print("Before GP Gradient: ",round(torch.cuda.memory_allocated(0)/1024**3,1))
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_input,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs = True 
        )[0]

        #print("GP Gradient: ",round(torch.cuda.memory_allocated(0)/1024**3,1))
        gradient = gradient.contiguous().view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        del alpha
        del interpolated_input
        del mixed_scores
        del gradient 
        del gradient_norm

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return gradient_penalty

    def create_epoch_batches(self, df_length, batch_size, shuffle=True, same_size_batching=False):
        """
        :param df_length: int
            total number of samples in training set
        :param batch_size: int
            number of samples in one atch
        :param shuffle:
            keep order of training set or random shuffle
        :return epoch: list of list
            list of listis containing indices for each batch for one epoch
        """
        if same_size_batching:
            epoch = []
            foundlings = []
            for length in self.sorted_length_keys:
                length_idxs = self.train_length_dict[length]
                rest = len(length_idxs) % batch_size
                random.shuffle(length_idxs)
                epoch += [length_idxs[i * batch_size:(i * batch_size) + batch_size] for i in
                          range(int(len(length_idxs) / batch_size))]
                if rest > 0:
                    foundlings += list(length_idxs[-rest:])
            foundlings = np.asarray(foundlings)
            rest = len(foundlings) % batch_size
            epoch += [foundlings[i * batch_size:(i * batch_size) + batch_size] for i in
                      range(int(len(foundlings) / batch_size))]
            if rest > 0:
                epoch += [foundlings[-rest:]]
            random.shuffle(epoch)

        else:
            rest = df_length % batch_size
            idxs = list(range(df_length))
            if shuffle:
                random.shuffle(idxs)
            if rest > 0:
                idxs += idxs[:(batch_size - rest)]
            epoch = [idxs[i * batch_size:(i * batch_size) + batch_size] for i in range(int(len(idxs) / batch_size))]

        return epoch


    def train(self, num_epochs,
              continue_training_from,
              critic_iteration_schedule,
              lambda_gp,
              plot_every_i_iterations=1,
              save_plot_after_i_iterations=1,
              save_model_after_i_iterations=1,
              shuffle=True,
              verbose=True,
              dict_file="",
              file_to_store= time.strftime("%Y%m%d-%H%M%S")):

        """
                Train the conditional gan models with Wasserstein Distance + Gradient Penalty

                :param num_epochs: int
                    number of epochs to train
                :param continue_training_from: int
                    epoch to resume training from
                :param critic_iteration_schedule: dict {n_critic_iterations : for_n_epochs}
                    dictionary containing the schedule of critic iterations
                :param lambda_gp: int
                    weighting hyperparameter for gradient penalty
                :param plot_every_i_iterations: int
                    number of epochs before plotting the results on the fixed batch
                :param save_plot_after_i_iterations: int
                    number of epochs before saving the plotted results
                :param save_model_after_i_iterations: int
                    number of epochs to train model before saving it
                :param shuffle: bool
                    whether to use shuffle in creating epochs
                :param verbose: bool
                    print results after each epoch
                :param dict_file: str
                    dictionary to store the model in
                :param file_to_store: str
                    name of files to store model, training and validation results
                """

        critic_iterations = list(critic_iteration_schedule.keys())
        critic_iterations_switch_points = np.cumsum(list(critic_iteration_schedule.values()))


        if continue_training_from > 0:
            print("Continue Training: iteration %d..." % continue_training_from)
            for i in range(continue_training_from):
                epoch = self.create_epoch_batches(len(self.inps),  self.batch_size,shuffle=shuffle,same_size_batching=self.use_same_size_batching)
                critic_iteration = critic_iterations[np.where(i < critic_iterations_switch_points)[0][0]]
                for jj, idxs in enumerate(epoch):
                    lens_input_jj = self.lens[idxs]
                    for _ in range(critic_iteration):
                        cur_batch_size = len(idxs)
                        noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in lens_input_jj])
                        #noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(lens_input_jj)])
                        del noise
    
                    noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in lens_input_jj])
                    #noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(lens_input_jj)])
                    del noise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Start Training... ")
        
        start_time = time.time()
        for ii in tqdm(range(num_epochs), desc='Training...', position=0, leave=True):
            ii += continue_training_from
            epoch = self.create_epoch_batches(len(self.inps), self.batch_size, shuffle=shuffle, same_size_batching=self.use_same_size_batching)

            loss_gen_list = []
            w_loss_list = []
            loss_critic_list = []
            critic_fake_list = []
            critic_real_list = []
            

            critic_iteration = critic_iterations[np.where(ii < critic_iterations_switch_points)[0][0]]
            for jj, idxs in enumerate(tqdm(epoch, desc="Batch...", position=1, leave=False)):     
                
                cur_time = time.time()
                
                lens_input_jj = self.lens[idxs]
                real = self.inps.iloc[idxs]
                real = pad_batch_online(lens_input_jj, real, self.device)
                vectors_jj = self.vectors[idxs]
                cur_batch_size = real.shape[0]
                cur_batch_length = real.shape[1]
                
                #print(cur_batch_length)
                # Train Critic: max E[critic(real)] - E[critic(fake)]
                # equivalent to minimizing the negative of that
                # with torch.backends.cudnn.flags(enabled=False): #https://github.com/facebookresearch/higher#knownpossible-issues
                if critic_iteration > 0:
                    for _ in range(critic_iteration):
                        #with torch.backends.cudnn.flags(enabled=False):
                        self.opt_critic.zero_grad()
                        #noise = torch.randn(cur_batch_size, 1, self.z_dim).to(self.device)
                        noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in lens_input_jj])
                        #noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(lens_input_jj)])
                        noise = pad_batch_online(lens_input_jj, noise, device = self.device)
                        #print("Critic Noise", round(torch.cuda.memory_allocated(0)/1024**3,1))
                        
                        with torch.backends.cudnn.flags(enabled=False):
                            fake = self.gen(noise, lens_input_jj, vectors_jj)
                            #print("Fake Generation", round(torch.cuda.memory_allocated(0)/1024**3,1))
                            
                            critic_real = self.critic(real, lens_input_jj, vectors_jj).reshape(-1)
                            critic_fake = self.critic(fake, lens_input_jj, vectors_jj).reshape(-1)
                            #print("Critic", round(torch.cuda.memory_allocated(0)/1024**3,1))
                            
                            #print("Critic Loss")
                            critic_diff = critic_real - critic_fake 
                            loss_diff = torch.mean(critic_diff)
                            #print("Critic Loss", round(torch.cuda.memory_allocated(0)/1024**3,1))
                            #cri_loss_fake = torch.mean(critic_fake)
                            #cri_loss_real = -torch.mean(critic_real)
                            #loss_diff = cri_loss_fake - cri_loss_real

                            # STD
                            # critic_diff_std = torch.std(critic_real) - torch.std(critic_fake)
                            gp = self.gradient_penalty(self.critic, lens_input_jj, vectors_jj, real, fake, device=self.device)
                            #print("GP", round(torch.cuda.memory_allocated(0)/1024**3,1))
                            was_loss = -loss_diff + lambda_gp * gp  # - critic_diff_std
                            #print("W-Loss", round(torch.cuda.memory_allocated(0)/1024**3,1))
                            # was_loss = -(loss_real - loss_fake) + LAMBDA_GP * gp
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            

                            #was_loss.backward(retain_graph=True)
                            was_loss.backward()
                            # loss_critic.backward()
                            self.opt_critic.step()
    
                            del noise 
                            del gp 
                            del fake 
                            del critic_diff
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            #print("Critic Iteration", round(torch.cuda.memory_allocated(0)/1024**3,1))
                    
                    loss_critic_list += [loss_diff.item()]
                    w_loss_list += [was_loss.item()]
                    del was_loss
                    del loss_diff
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    
                # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                self.opt_gen.zero_grad()
                #noise = torch.randn(cur_batch_size, 1, self.z_dim).to(self.device)
                
                noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in lens_input_jj])
                #noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(lens_input_jj)])
                noise = pad_batch_online(lens_input_jj, noise, device = self.device)
                
                with torch.backends.cudnn.flags(enabled=False):
                    fake = self.gen(noise, lens_input_jj, vectors_jj)
                    #print("Fake Generation", round(torch.cuda.memory_allocated(0)/1024**3,1))
                    
                    critic_fake = self.critic(fake, lens_input_jj, vectors_jj).reshape(-1)
                    critic_real = self.critic(real, lens_input_jj, vectors_jj).reshape(-1)
                    #print("Critic Evaluation", round(torch.cuda.memory_allocated(0)/1024**3,1))
                    
                    #critic_diff = critic_fake - critic_real
                    velocity_loss, jerk_loss = velocity_jerk_loss(fake, rmse_loss)
                    loss_gen = -torch.mean(critic_fake) + velocity_loss + jerk_loss  # + critic_diff_std
                    # critic_diff_std = torch.std(critic_real) - torch.std(gen_fake)
                    #print("Generator Loss", round(torch.cuda.memory_allocated(0)/1024**3,1))

                    # loss_gen = -torch.mean(gen_fake)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    loss_gen.backward()
                    self.opt_gen.step()

                    del noise
                    del real 
                    del fake 
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                
                
                
                loss_gen_list += [loss_gen.item()] 
                critic_fake_list += [torch.mean(critic_fake).item()]
                critic_real_list += [torch.mean(critic_real).item()]

                del loss_gen
                del critic_fake
                del critic_real 
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
                #print("Append Loss", round(torch.cuda.memory_allocated(0)/1024**3,1))

                for p in self.critic.parameters(): # clear gradients to reduce storage
                    p.grad = None
                for p in self.gen.parameters():
                    p.grad = None

                if verbose:
                    if (jj + 1) % 50 == 0:
                        print(f"Epoch [{ii}/{num_epochs+continue_training_from}] Batch {jj + 1}/{len(epoch)}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                        
            cur_time = time.time() - cur_time


            mean_loss_critic = np.mean(loss_critic_list)
            mean_loss_gen = np.mean(loss_gen_list)
            mean_loss_was = np.mean(w_loss_list)
            

            self.res_train.loc[self.res_train_ix] = [ii, mean_loss_critic, mean_loss_gen, mean_loss_was]
            self.res_train_ix += 1

            if verbose:
                print(f"Epoch [{ii}/{num_epochs+continue_training_from}]  \
                              Loss Critic: {mean_loss_critic:.4f},Loss Generator: {mean_loss_gen:.4f},  Wasser Loss: {mean_loss_was:.4f}")
                print('Time Taken: {:.4f} seconds. Estimated {:.4f} hours remaining'.format(cur_time, ((num_epochs+continue_training_from)-ii)*(cur_time)/3600))
                print(f'Memory Usage:{ii}')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

            if (ii+1) % plot_every_i_iterations == 0:
                if self.tgt_name == "mel":
                    plot_fake_mels(self.gen, self.fixed_noise, self.fixed_reals, self.fixed_vectors,self.fixed_labels,
                                   ii,save_plot_after_i_iterations,
                                   dict_file,file_to_store,starting_i = 0)

                else:
                    if self.plot_mean_comparison_to is None:
                        colors = ["C%d" % i for i in range(5)]

                        plot_fake_cps(self.gen,self.fixed_noise, self.fixed_reals, self.fixed_vectors,self.fixed_labels,
                                      ii, save_plot_after_i_iterations,
                                      dict_file, file_to_store, colors)
                    else:
                        plot_cp_mean_comparison(self.gen, self.fixed_noise, self.fixed_reals,
                                                self.fixed_lens,
                                                self.fixed_vectors, self.plot_mean_comparison_to,
                                                ii, save_plot_after_i_iterations, dict_file, file_to_store,
                                                device=self.device)
            
            if (ii + 1) % save_model_after_i_iterations == 0:
                self.res_train.to_pickle(
                    dict_file + "/res_train_" + file_to_store + "_%d" % (ii + 1) + ".pkl")
                with open(dict_file + "/generator_"+ file_to_store + "_%d" % (ii + 1) + ".pkl",
                          "wb") as pfile:
                    pickle.dump((self.gen, self.opt_gen), pfile)
                with open(dict_file + "/critic_" + file_to_store + "_%d" % (ii + 1) + ".pkl", "wb") as pfile:
                    pickle.dump((self.critic, self.opt_critic), pfile)
                    
    def pre_evaluate(self):
        self.gen.eval()
        valid_predictions = []
        valid_losses = []
        valid_sublosses = []

        with torch.no_grad():  # no gradient calculation
            epoch_valid = self.create_epoch_batches(len(self.inps_valid), 1, shuffle=False)
            for jj, idxs in enumerate(epoch_valid):
                lens_input_jj = self.lens_valid[idxs]
                vectors_jj = self.vectors_valid[idxs]

                noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in lens_input_jj])
                #noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(lens_input_jj)])
                noise = pad_batch_online(lens_input_jj, noise, device = self.device)

                batch_real = self.inps_valid.iloc[idxs]
                batch_real = pad_batch_online(lens_input_jj, batch_real,self.device)
                batch_real = torch.squeeze(batch_real,1)


                Y_hat = self.gen(noise,lens_input_jj,vectors_jj)

                loss = self.pre_criterion(Y_hat, batch_real)

                if isinstance(loss, tuple): # sublosses
                    sub_losses = loss[1:]  # rest sublosses
                    loss = loss[0] # first total loss

                    valid_losses += [loss.item()]
                    valid_sublosses += [[sub_loss.item() for sub_loss in sub_losses]] # for each samples [subloss1_i,subloss2_i,subloss3_i]

                else:
                    valid_losses += [loss.item()]

                prediction = Y_hat.cpu().detach().numpy()[0]
                valid_predictions +=[prediction]

            if len(valid_sublosses) > 0:
                test_sublosses = np.asarray(valid_sublosses)
                return valid_predictions, valid_losses, \
                       [test_sublosses[:, i] for i in range(test_sublosses.shape[1])] # for each subloss [subloss1_i, subloss1_j,subloss1_k]

            else:
                return valid_predictions, valid_losses, []

                    
    def pre_train(self, num_epochs,
              continue_training_from,
              validate_after_i_epochs=1,    
              save_model_after_i_iterations=1,
              shuffle=True,
              verbose=True,
              dict_file="",
              file_to_store= time.strftime("%Y%m%d-%H%M%S")):
        
        if continue_training_from > 0:
            #print("Continue Pre-Training: iteration %d..." % continue_training_from)
            epoch_valid = self.create_epoch_batches(len(self.inps_valid), 1, shuffle=False)
            
            for jj, idxs in enumerate(epoch_valid): 
                lens_input_jj = self.lens_valid[idxs]        
                noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in lens_input_jj])
                #noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(lens_input_jj)])
                #noise = pad_batch_online(lens_input_jj, noise, device = self.device)
            del noise
            
            for i in tqdm(range(continue_training_from), desc='Continue Pre-Training: iteration %d...'):
                epoch = self.create_epoch_batches(len(self.inps),  self.pre_batch_size,shuffle=shuffle,same_size_batching=self.use_same_size_batching)
                epoch_valid = self.create_epoch_batches(len(self.inps_valid), 1, shuffle=False)
                for jj, idxs in enumerate(epoch):
                    lens_input_jj = self.lens[idxs] 
                    noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in lens_input_jj])
                    #noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(lens_input_jj)])
                
                for jj, idxs in enumerate(epoch_valid):    
                    lens_input_jj = self.lens_valid[idxs]        
                    noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in lens_input_jj])
                    #noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(lens_input_jj)])
                    #noise = pad_batch_online(lens_input_jj, noise, device = self.device)
                del noise
                    
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Start Pre-Training... ")
            _, validation_losses, validation_sublosses = self.pre_evaluate()
            
            if len(validation_sublosses) > 0:
                average_epoch_valid_sublosses = [np.mean(sub_loss) for sub_loss in validation_sublosses] # calculate mean sublosses
                self.pre_res_valid.loc[self.pre_res_valid_ix] = [-1, np.mean(validation_losses)] + average_epoch_valid_sublosses + [param_group["lr"] for param_group in self.pre_opt_gen.param_groups]
            else:
                average_epoch_valid_sublosses = []
                self.pre_res_valid.loc[self.pre_res_valid_ix] = [-1, np.mean(validation_losses)] + [param_group["lr"] for param_group in self.pre_opt_gen.param_groups]
            self.pre_res_valid_ix += 1

            if verbose:
                print("\nInitial Validation Loss: ", np.mean(validation_losses))
                if len(average_epoch_valid_sublosses) > 0:
                    for i, subloss in enumerate(average_epoch_valid_sublosses):
                        loss_name = list(self.pre_res_valid.columns)[2+i]
                        print(f"Subloss {loss_name}: {subloss}")
            
        
        start_time = time.time()
        for ii in tqdm(range(num_epochs), desc='Training...', position=0, leave=True):
            self.gen.train()
            ii += continue_training_from
            average_epoch_loss = []
            average_epoch_sublosses = []

            running_loss = 0.0
            epoch = self.create_epoch_batches(len(self.inps), self.pre_batch_size, shuffle=shuffle, same_size_batching=self.use_same_size_batching)
            for jj, idxs in enumerate(epoch):
                lens_input_jj = self.lens[idxs]
                real = self.inps.iloc[idxs]
                real = pad_batch_online(lens_input_jj, real, self.device)
               
                real = torch.squeeze(real,1)
                vectors_jj = self.vectors[idxs]
                
                noise = pd.Series([torch.randn(int(l.item()), self.z_dim) for l in lens_input_jj])
                #noise = pd.Series([np.asarray([sample_multivariate_normal(self.noise_mean, self.decomposition_matrix, self.decomposition) for x in range(int(l.item()))]) for _,l in enumerate(lens_input_jj)])
                noise = pad_batch_online(lens_input_jj, noise, device = self.device)
                
                Y_hat = self.gen(noise,lens_input_jj,vectors_jj)
                
                self.pre_opt_gen.zero_grad()
                loss = self.pre_criterion(Y_hat, real)
                if isinstance(loss, tuple):
                    sub_losses = loss[1:]  # sublosses
                    loss = loss[0] # total loss

                    average_epoch_loss += [loss.item()]
                    average_epoch_sublosses += [[sub_loss.item() for sub_loss in sub_losses]] # for each batch [subloss1_i,subloss2_i,subloss3_i]

                else:
                    average_epoch_loss += [loss.item()]

                running_loss += loss.item()
                loss.backward()  # compute dloss/dx and accumulated into x.grad
                self.pre_opt_gen.step()  # compute x += -learning_rate * x.grad

            if len(average_epoch_sublosses) > 0:
                average_epoch_sublosses = np.asarray(average_epoch_sublosses)
                average_epoch_sublosses = [average_epoch_sublosses[:, i] for i in
                                           range(average_epoch_sublosses.shape[1])]  # for each subloss [subloss1_i, subloss1_j,subloss1_k]
                average_epoch_sublosses = [np.mean(sub_loss) for sub_loss in average_epoch_sublosses]
                self.pre_res_train.loc[self.pre_res_train_ix] = [ii, np.mean(average_epoch_loss)] + average_epoch_sublosses + [param_group["lr"] for param_group in self.pre_opt_gen.param_groups]
            else:
                average_epoch_sublosses = []
                self.pre_res_train.loc[self.pre_res_train_ix] = [ii, np.mean(average_epoch_loss)] + [param_group["lr"] for param_group in self.pre_opt_gen.param_groups]

            self.pre_res_train_ix += 1

            if verbose:
                print("\n---------------- Trainin Epoch %d ----------------" % (ii+1))
                print("Avg Training Loss: ", np.mean(average_epoch_loss))
                print("Running Training Loss: ", float(running_loss))
                if len(average_epoch_sublosses) > 0:
                    for i, subloss in enumerate(average_epoch_sublosses):
                        loss_name = list(self.pre_res_train.columns)[2+i]
                        print(f"Subloss {loss_name}: {subloss}")

            ########################################################
            ###################### Validation ######################
            ########################################################

            if (ii+1) % validate_after_i_epochs == 0:
                _, validation_losses, validation_sublosses = self.pre_evaluate()
                if len(validation_sublosses) > 0:
                    average_epoch_valid_sublosses = [np.mean(sub_loss) for sub_loss in validation_sublosses]
                    self.pre_res_valid.loc[self.pre_res_valid_ix] = [ii, np.mean(
                        validation_losses)] + average_epoch_valid_sublosses + [param_group["lr"] for param_group in
                                                                               self.pre_opt_gen.param_groups]
                else:
                    average_epoch_valid_sublosses = []
                    self.pre_res_valid.loc[self.pre_res_valid_ix] = [ii, np.mean(validation_losses)] + [param_group["lr"] for param_group in self.pre_opt_gen.param_groups]
                self.pre_res_valid_ix += 1

                if verbose:
                    print("\nAvg Validation Loss ", np.mean(validation_losses))
                    if len(average_epoch_valid_sublosses) > 0:
                        for i, subloss in enumerate(average_epoch_valid_sublosses):
                            loss_name = list(self.pre_res_valid.columns)[2+i]
                            print(f"Subloss {loss_name}: {subloss}")

                if not save_model_after_i_iterations is None:
                    if ii > 0 and (ii+1) % save_model_after_i_iterations == 0:
                        self.pre_res_train.to_pickle(dict_file + "/pre_res_train_" + file_to_store + "_%d" % (ii+1) + ".pkl")
                        self.pre_res_valid.to_pickle(dict_file + "/pre_res_valid_" + file_to_store + "_%d" % (ii+1) + ".pkl")
                        with open(dict_file + "/pre_trained_generator_" + file_to_store + "_%d" % (ii+1) + ".pkl", "wb") as pfile:
                            pickle.dump((self.gen, self.pre_opt_gen), pfile)

                
