import torch.optim as optim
from Loss import *
from scipy.io import loadmat, savemat

def Train_Longitudinal(model, group_data, individual_data_dict, steps, log_interval, save_model_path, device='cuda:0'):

    ages = list(individual_data_dict.keys())

    optimizers = optim.AdamW(model.encoders.parameters(), lr=0.01)
    schedulers = optim.lr_scheduler.StepLR(optimizers, step_size=500, gamma=0.5)

    group_data = group_data.to(device)
    for age in ages:
        individual_data_dict[age] = individual_data_dict[age].to(device)

    model.to(device)

    Parcellation_sim_Loss1_dict={}
    Parcellation_homo_FCN_Loss2_dict = {}
    Parcellation_homo_MSN_Loss2_dict = {}
    Spatial_cont_Loss3_dict = {}

    individual_FCN_feature_dict={}
    for age in individual_data_dict.keys():
        individual_FCN_feature_dict[str(age)] = individual_data_dict[str(age)].x[:group_data.x.shape[0] // 2]
    individual_MSN_feature_dict={}
    for age in individual_data_dict.keys():
        individual_MSN_feature_dict[str(age)] = individual_data_dict[str(age)].x[group_data.x.shape[0] // 2:]
    Feature_Consistency_FCN_Loss4 = Feature_Consistency_Loss(group_data.y,group_data.x[:group_data.x.shape[0] // 2],individual_FCN_feature_dict,age_list=model.ages)
    Feature_Consistency_MSN_Loss4 = Feature_Consistency_Loss(group_data.y,group_data.x[group_data.x.shape[0] // 2:], individual_MSN_feature_dict,age_list=model.ages)
    Longitudinal_loss_loss6 = Longitudinal_Consistency_Loss(model.ages)

    epoch_loss_sim = {}
    epoch_loss_cont = {}
    epoch_loss_homo = {}
    epoch_FCN_parcellation_homo = {}
    epoch_MSN_parcellation_homo = {}
    epoch_loss_longitudinal=0
    epoch_loss_feature=0

    for step in range(steps):
        model.train()

        optimizers.zero_grad()

        assignment_outputs_dict = model(individual_data_dict)

        loss_longitudinal = Longitudinal_loss_loss6(assignment_outputs_dict)

        loss_feature_cosistence = (Feature_Consistency_FCN_Loss4(assignment_outputs_dict) + Feature_Consistency_MSN_Loss4(assignment_outputs_dict))/2

        epoch_loss_longitudinal += loss_longitudinal.item()

        epoch_loss_feature += loss_feature_cosistence.item()

        total_combined_loss = 0

        for age in ages:

            individual_data = individual_data_dict[age]
            individual_assignment_output = assignment_outputs_dict[age]

            if step==0:

                Parcellation_sim_Loss1_dict[age] = Parcellation_Similarity_Loss(group_data.y)

                Parcellation_homo_FCN_Loss2_dict[age] = Dynamic_Spatial_Homogeneity_Loss(group_data.edge_index, individual_data.x[:individual_data.x.shape[0] // 2], individual_data.pos, parcel_num=model.encoders.num_class, smooth_weight=0.25)
                Parcellation_homo_MSN_Loss2_dict[age] = Dynamic_Spatial_Homogeneity_Loss(group_data.edge_index, individual_data.x[individual_data.x.shape[0] // 2:], individual_data.pos, parcel_num=model.encoders.num_class, smooth_weight=0.25)

                Spatial_cont_Loss3_dict[age] = Spatial_Continuity_Loss(group_data.edge_index)


            loss_sim = Parcellation_sim_Loss1_dict[age](individual_assignment_output)

            loss_homo_FCN, FCN_parcellation_homo = Parcellation_homo_FCN_Loss2_dict[age](individual_assignment_output)
            loss_homo_MSN, MSN_parcellation_homo = Parcellation_homo_MSN_Loss2_dict[age](individual_assignment_output)
            loss_homo = (loss_homo_FCN + loss_homo_MSN) / 2

            loss_cont = Spatial_cont_Loss3_dict[age](individual_assignment_output)

            if step==0:
                epoch_loss_sim[age]=loss_sim.item()
                epoch_loss_cont[age] = loss_cont.item()
                epoch_loss_homo[age] = loss_homo.item()
                epoch_FCN_parcellation_homo[age] = FCN_parcellation_homo
                epoch_MSN_parcellation_homo[age] = MSN_parcellation_homo
            else:
                epoch_loss_sim[age]+=loss_sim.item()
                epoch_loss_cont[age] += loss_cont.item()
                epoch_loss_homo[age] += loss_homo.item()
                epoch_FCN_parcellation_homo[age] += FCN_parcellation_homo
                epoch_MSN_parcellation_homo[age] += MSN_parcellation_homo

            if step<=60:

                total_combined_loss += 2 * loss_sim + 40 * loss_cont

            else:

                total_combined_loss += 2 * loss_sim + 40 * loss_cont + 60 * loss_homo

        if step>60:

            total_combined_loss += 40 * loss_longitudinal + 40 * loss_feature_cosistence

        total_combined_loss.backward()

        optimizers.step()
        schedulers.step()

        if step % 10 == 0 and step > 60:
            with torch.no_grad():
                for age in ages:
                    Parcellation_homo_FCN_Loss2_dict[age].update_tau(assignment_outputs_dict[age])
                    Parcellation_homo_MSN_Loss2_dict[age].update_tau(assignment_outputs_dict[age])

        if step % log_interval == 0 and step > 0:

            with torch.no_grad():
                epoch_loss_longitudinal = epoch_loss_longitudinal / log_interval
                epoch_loss_feature = epoch_loss_feature / log_interval
                for age in ages:
                    epoch_loss_sim[age] = epoch_loss_sim[age] / log_interval
                    epoch_loss_cont[age] = epoch_loss_cont[age] / log_interval
                    epoch_loss_homo[age] = epoch_loss_homo[age] / log_interval
                    epoch_FCN_parcellation_homo[age] = epoch_FCN_parcellation_homo[age] / log_interval
                    epoch_MSN_parcellation_homo[age] = epoch_MSN_parcellation_homo[age] / log_interval

                    print(
                        f'Age:{age}: Epoch [{step}/ {steps}]: epoch_loss_sim:{epoch_loss_sim[age]:.4f}; epoch_loss_cont:{epoch_loss_cont[age]:.4f}; epoch_loss_homo:{epoch_loss_homo[age]:.4f}; FCN_homo:{epoch_FCN_parcellation_homo[age]:.4f}; MSN_homo:{epoch_MSN_parcellation_homo[age]:.4f}; longitudinal_loss:{epoch_loss_longitudinal:.4f}; feature_consistence_loss:{epoch_loss_feature:.4f}')

                    epoch_loss_sim[age] = 0
                    epoch_loss_cont[age] = 0
                    epoch_loss_homo[age] = 0
                    epoch_FCN_parcellation_homo[age] = 0
                    epoch_MSN_parcellation_homo[age] = 0

                epoch_loss_longitudinal=0
                epoch_loss_feature=0

    torch.save(model.state_dict(), save_model_path)

def Test_Longitudinal(args,model, individual_data_dict, save_parcellation_path, device='cuda:0'):

    model.eval()

    ages = list(individual_data_dict.keys())

    for age in ages:
        individual_data_dict[age] = individual_data_dict[age].to(device)

    model.to(device)

    assignment_outputs_dict = model(individual_data_dict)

    for age in ages:

        individual_assignment_output = assignment_outputs_dict[age]

        individual_assignment_output=F.softmax(individual_assignment_output).detach().cpu().numpy()

        predict_label = np.argmax(individual_assignment_output,axis=1,keepdims=True)

        medial_wall_path = args.medial_wall_path
        medial_wall_data = loadmat(medial_wall_path)
        medial_wall_data_arr = medial_wall_data['medial_lh']

        final_label = np.zeros(shape=(medial_wall_data_arr.shape[0], 1))
        final_label[medial_wall_data_arr == 1] = predict_label.flatten()+1
        savemat(f'{save_parcellation_path}/individual_parcellation_{age}.mat', {'label_lh': final_label})