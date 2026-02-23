from tqdm import tqdm
import torch
import torchrl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping, MRFLoss, OldMRFLoss
import matplotlib.pyplot as plt
import pandas as pd
from dataset import Vibrate, Mask


def imagen_instantblur(model, label, device, params):
    #image  = model.image.emission.sample(label, params)
    out    = model.image.blur(label)
    image  = out["out"]
    image  = model.image.noise(image)
    image  = model.image.preprocess.sample(image)
    return image

def imagen_instantblur_without_noise(model, label, device, params):
    out    = model.image.blur(label)
    image  = out["out"]
    image  = model.image.preprocess.sample(image)
    return image

def _loss_fnz(_input, mask, target):
    if mask is not None:
        label_loss = F.gaussian_nll_loss(
            input  = _input*mask            , 
            target = target                 ,
            var    = torch.ones_like(_input) )
    else:
        label_loss = 0.
    trnorm   = torchrl.modules.TruncatedNormal(
        loc     = torch.zeros_like(_input),
        scale   = torch.ones_like(_input) ,
        upscale = torch.zeros_like(_input) )
    log_prob = trnorm.log_prob(torch.log(_input))
    nll_loss = - torch.mean(log_prob) * 1/100
    return label_loss + nll_loss

def _loss_fnx(_input, target, a, sigma):
    """
    gaussian nll loss for poisson noise
    """
    a     = torch.max(a, 1e-3*torch.ones_like(a))
    sigma = torch.max(sigma, 1e-3*torch.ones_like(sigma))
    var   = torch.clamp_min(a * _input + sigma, min=0.)
    loss  = F.gaussian_nll_loss(_input, target, var)
    return loss

def pretrain_loop(
        optimizer            ,
        model                ,
        train_loader         ,
        val_loader           ,
        model_name           ,
        params               ,
        train_loop_params    ,
        train_dataset_params ,
        vibration_params     ,
        scheduler            ,
        without_noise        ,
        ):
    
    n_epochs     = train_loop_params["n_epochs"]  
    device       = params["device"]                               
    path         = train_loop_params["path"]            
    savefig_path = train_loop_params["savefig_path"]
    es_patience  = train_loop_params["es_patience"]
    is_vibrate   = train_loop_params["is_vibrate"]
    loss_fnx     = eval(train_loop_params["loss_fnx"])
    wx           = train_loop_params["weight_x"]        
    wz           = train_loop_params["weight_z"]

    earlystopping = EarlyStopping(
        name        = model_name ,
        path        = path       ,
        patience    = es_patience,
        window_size = 3          ,
        metric      = "mean"     ,
        verbose     = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    train_curve = pd.DataFrame(columns=["training loss", "validatation loss"])
    loss_list, vloss_list = [], []
    vibrate = Vibrate(vibration_params)
    mask = Mask()
    for epoch in range(1, n_epochs + 1):
        loss_sum = 0.
        model.train()
        for train_data in train_loader:
            labelx = train_data["labelx"].to(device = device)
            labelz = train_data["labelz"].to(device = device)
            with torch.no_grad():
                if without_noise:
                    image = imagen_instantblur_without_noise(
                        model  = model ,
                        label  = labelz,
                        device = device,
                        params = params,)
                else:
                    image = imagen_instantblur(
                        model  = model ,
                        label  = labelz,
                        device = device,
                        params = params,)
                image  = model.image.hill.sample(image)
            vimage = vibrate(image) if is_vibrate else image
            vimage = mask.apply_mask(
                train_dataset_params["mask"]      ,
                vimage                            ,
                train_dataset_params["mask_size"] ,
                train_dataset_params["mask_num"]  ,)
            outdict = model(vimage)
            out   = outdict["enhanced_image" ]
            lum   = outdict["estim_luminance"]
            lossx  = loss_fnx(out, labelx)
            lossz  = _loss_fnz(
                _input = lum    ,
                mask   = labelx ,
                target = labelz  )
            loss = wx * lossx + wz * lossz
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            loss_sum += loss.detach().item()
        
        vloss_sum = 0.
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                labelx = val_data["labelx"].to(device = device)
                labelz = val_data["labelz"].to(device = device)
                if without_noise:
                    image = imagen_instantblur_without_noise(
                        model  = model ,
                        label  = labelz,
                        device = device,
                        params = params,)
                else:
                    image = imagen_instantblur(
                        model  = model ,
                        label  = labelz,
                        device = device,
                        params = params,)
                image  = model.image.hill.sample(image)
                vimage = vibrate(image) if is_vibrate else image
                outdict = model(vimage)
                out   = outdict["enhanced_image" ]
                lum   = outdict["estim_luminance"]
                lossx = loss_fnx(out, labelx)
                lossz = _loss_fnz(lum, labelx, labelz)
                vloss = wx * lossx + wz * lossz
                vloss_sum += vloss.detach().item()
        
        num  = len(train_loader)
        vnum = len(val_loader)
        loss_list.append(loss_sum / num)
        vloss_list.append(vloss_sum / vnum)
        writer.add_scalar('train loss', loss_sum / num, epoch)
        writer.add_scalar('val loss', vloss_sum / vnum, epoch)
        row = pd.DataFrame([[loss_list[-1], vloss_list[-1]]],
                           columns = train_curve.columns)
        train_curve = pd.concat([train_curve, row], ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv",
                           index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, ' + \
                  f'Val {vloss_list[-1]}')
            
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])

        vibrate.step()
        if vibrate.num_step >= vibrate.max_step:
            condition = True
        else:
            condition = False

        earlystopping((vloss_sum / vnum), model, optimizer, condition = condition)
        #if earlystopping.early_stop:
        #    break
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    plt.legend()
    plt.savefig(
        f'{savefig_path}/{model_name}_train.png', format='png', dpi=500)

def luminance_adjustment(rec, image):
    e = 1e-7
    cov = torch.mean((rec - torch.mean(rec)) * (image - torch.mean(image)))
    var = (torch.mean((rec - torch.mean(rec)) ** 2))
    beta  = (cov + e) / (var + e)
    beta  = torch.clip(beta, min=0., max=30.)           
    alpha = torch.mean(image) - beta * torch.mean(rec)  
    return alpha + beta * rec

def finetuning_loop(
        optimizer            ,
        model                ,
        train_loader         ,
        val_loader           ,
        device               ,
        model_name           ,
        ewc                  ,
        train_dataset_params ,
        train_loop_params    ,
        vibration_params     ,
        scheduler = None     ,
        v_verbose = False    ,
        ):
    n_epochs         = train_loop_params["n_epochs"         ]
    path             = train_loop_params["path"             ]
    savefig_path     = train_loop_params["savefig_path"     ]
    adjust_luminance = train_loop_params["adjust_luminance" ]
    es_patience      = train_loop_params["es_patience"      ]
    is_vibrate       = train_loop_params["is_vibrate"       ]
    zloss_weight     = train_loop_params["zloss_weight"     ]
    ewc_weight       = train_loop_params["ewc_weight"       ]
    qloss_weight     = train_loop_params["qloss_weight"     ]
    ploss_weight     = train_loop_params["ploss_weight"     ]
    mrfloss_order    = train_loop_params["mrfloss_order"    ]
    mrfloss_weights  = train_loop_params["mrfloss_weights"  ]
    dilation         = train_loop_params["mrfloss_dilation" ]
    
    earlystopping = EarlyStopping(
        name        = model_name ,
        path        = path       ,
        patience    = es_patience,
        window_size = 1          ,
        metric      = "mean"     ,
        verbose     = True        )
    writer = SummaryWriter(f'runs/{model_name}')
    train_curve = pd.DataFrame(columns=["training loss"    ,
                                        "validatation loss"] )
    loss_list, vloss_list= [], []
    vibrate = Vibrate(vibration_params)
    mask = Mask()
    train_with_mrf = True
    mrf_loss =\
    MRFLoss(
        dims     = 3               ,
        order    = mrfloss_order   ,
        weights  = mrfloss_weights ,
        dilation = dilation        ,)
    #OldMRFLoss(dims=3,
    #           order = mrfloss_order,
    #           mode="all")

    for epoch in range(1, n_epochs + 1):
        loss_sum    = 0.
        xloss_sum   = 0.
        zloss_sum   = 0.
        mrfloss_sum = 0.
        qloss_sum   = 0.
        ploss_sum   = 0.
        ewcloss_sum = 0.
        model.train()
        for train_data in tqdm(train_loader,
                               bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            image  = train_data["image"].to(device = device)
            #_image = model.image.hill.sample(image)
            _image  = model.image.hill.sample(image)
            _image = mask.apply_mask(
                train_dataset_params["mask"]      ,
                _image                            ,
                train_dataset_params["mask_size"] ,
                train_dataset_params["mask_num"]  ,)
            vimage = vibrate(_image) if is_vibrate else _image
            outdict = model(vimage)
            rec     = outdict["reconstruction" ]
            a       = outdict["poisson_weight" ]
            sigma   = outdict["gaussian_sigma" ]
            out     = outdict["enhanced_image" ]
            lum     = outdict["estim_luminance"]
            qloss   = outdict["quantized_loss" ]
            ploss   = outdict["psf_loss"       ]
            if adjust_luminance:
                rec = luminance_adjustment(rec, image)
            #loss  = loss_fn(rec, image) * loss_weight
            loss = _loss_fnx(rec, image, a, sigma)
            xloss_sum += loss.detach().item()
            loss_z = _loss_fnz(
                _input = lum ,
                mask   = None,
                target = None ) * zloss_weight
            loss += loss_z
            zloss_sum += loss_z.detach().item()
            if train_with_mrf:
                loss_mrf = mrf_loss(out)
                loss += loss_mrf
                mrfloss_sum = loss_mrf.detach().item()
            if ewc is not None:
                ewc_loss = ewc.calc_ewc_loss(ewc_weight)
                loss += ewc_loss
                ewcloss_sum += ewc_loss.detach().item()
            if qloss is not None:
                qloss = qloss * qloss_weight
                loss += qloss
                qloss_sum += qloss.detach().item()

            if ploss is not None:
                ploss = ploss * ploss_weight
                loss += ploss
                ploss_sum += ploss.detach().item()
            loss_sum += loss.detach().item()
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        vloss_sum    = 0.
        vxloss_sum   = 0.
        vzloss_sum   = 0.
        vmrfloss_sum = 0.
        vqloss_sum   = 0.
        vploss_sum   = 0.
        vewcloss_sum = 0.
        model.eval()
        with torch.no_grad():
            for val_data in tqdm(val_loader,
                                 bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
                image   = val_data["image"].to(device = device)
                #_image  = model.image.hill.sample(image)
                _image  = model.image.hill.sample(image)
                vimage  = vibrate(_image) if is_vibrate else _image
                outdict = model(vimage)
                rec     = outdict["reconstruction" ]
                a       = outdict["poisson_weight" ]
                sigma   = outdict["gaussian_sigma" ]
                out     = outdict["enhanced_image" ]
                lum     = outdict["estim_luminance"]
                qloss   = outdict["quantized_loss" ]
                ploss   = outdict["psf_loss"       ]
                if adjust_luminance:
                    rec = luminance_adjustment(rec, image)
                #vloss   = loss_fn(rec, image) * loss_weight
                vloss = _loss_fnx(rec, image, a, sigma)
                vxloss_sum += vloss.detach().item()
                vloss = vloss.detach().item()             
                vloss_z = _loss_fnz(
                _input =lum ,
                mask   =None,
                target =None ) * zloss_weight
                vloss += vloss_z
                vzloss_sum += vloss_z.detach().item()
                vloss_sum += vloss.detach().item()
                if train_with_mrf:
                    loss_mrf = mrf_loss(out).detach().item()
                    vloss_sum += loss_mrf
                    vmrfloss_sum += loss_mrf
                if qloss is not None:
                    qloss = qloss.detach().item() * qloss_weight
                    vloss += qloss
                    vloss_sum += qloss
                    vqloss_sum += qloss
                if ploss is not None:
                    ploss = ploss.detach().item() * ploss_weight
                    vloss += ploss
                    vloss_sum += ploss
                    vploss_sum += ploss
                if ewc is not None:
                    ewc_loss = ewc.calc_ewc_loss(
                        ewc_weight).detach().item()
                    vloss_sum += ewc_loss
                    vloss += ewc_loss
                    vewcloss_sum += ewc_loss

        num  = len(train_loader)
        vnum = len(val_loader)
        loss_list.append(loss_sum / num)
        vloss_list.append(vloss_sum / vnum)
        writer.add_scalar('train loss'     , loss_sum     /  num  , epoch)
        writer.add_scalar('train x loss'   , xloss_sum    /  num  , epoch)
        writer.add_scalar('train z loss'   , zloss_sum    /  num  , epoch)
        writer.add_scalar('train mrf loss' , mrfloss_sum  /  num  , epoch)
        writer.add_scalar('train p loss'   , ploss_sum    /  num  , epoch)
        writer.add_scalar('train q loss'   , qloss_sum    /  num  , epoch)
        writer.add_scalar('train ewc loss' , ewcloss_sum  /  num  , epoch)
        writer.add_scalar('val loss'    , vloss_sum    / vnum  , epoch)
        writer.add_scalar('val x loss'  , vxloss_sum   / vnum  , epoch)
        writer.add_scalar('val z loss'  , vzloss_sum   / vnum  , epoch)
        writer.add_scalar('val mrf loss', vmrfloss_sum / vnum  , epoch)
        writer.add_scalar('val p loss'  , vploss_sum   / vnum  , epoch)
        writer.add_scalar('val q loss'  , vqloss_sum   / vnum  , epoch)
        writer.add_scalar('val ewc loss', vewcloss_sum / vnum  , epoch)
        writer.add_scalar('a'           , a.detach().item()    , epoch)
        writer.add_scalar('sigma'       , sigma.detach().item(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)
        row = pd.DataFrame([[loss_list[-1], vloss_list[-1]]],
                           columns = train_curve.columns)
        train_curve = pd.concat([train_curve, row], ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv",
                           index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]},'+\
                  f' Val {vloss_list[-1]}')
        vibrate.step()
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])
        condition = True
        earlystopping(vloss_list[-1], model, optimizer, condition = condition)

        if vloss_list[-1] == float('nan'):
            print('nan detected in validation loss. terminating...')
            break

        #if earlystopping.early_stop:
        #    break
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png',
                format='png', dpi=500)
    
def get_vibrate_condition(vibrateclass):
    return vibrateclass.num_step >= vibrateclass.max_step


def finetuning_see_loss(
        optimizer            ,
        model                ,
        train_loader         ,
        val_loader           ,
        device               ,
        model_name           ,
        ewc                  ,
        train_dataset_params ,
        train_loop_params    ,
        vibration_params     ,
        scheduler = None     ,
        v_verbose = False    ,
        ):
    
    n_epochs         = train_loop_params["n_epochs"         ]        
    path             = train_loop_params["path"             ]            
    savefig_path     = train_loop_params["savefig_path"     ]    
    adjust_luminance = train_loop_params["adjust_luminance" ]
    es_patience      = train_loop_params["es_patience"      ]
    is_vibrate       = train_loop_params["is_vibrate"       ]      
    zloss_weight     = train_loop_params["zloss_weight"     ]    
    ewc_weight       = train_loop_params["ewc_weight"       ]      
    qloss_weight     = train_loop_params["qloss_weight"     ]    
    ploss_weight     = train_loop_params["ploss_weight"     ]
    mrfloss_order    = train_loop_params["mrfloss_order"    ]
    mrfloss_weights  = train_loop_params["mrfloss_weights"  ]
    dilation         = train_loop_params["mrfloss_dilation" ]
    
    train_curve = pd.DataFrame(columns=["training loss"    ,
                                        "validatation loss"] )
    loss_list, vloss_list= [], []
    vibrate = Vibrate(vibration_params)
    mask = Mask()
    train_with_mrf = True
    mrf_loss =\
    MRFLoss(
        dims     = 3               ,
        order    = mrfloss_order   ,
        weights  = mrfloss_weights ,
        dilation = dilation        ,)
    #OldMRFLoss(dims=3,
    #           order = mrfloss_order,
    #           mode="all")

    for epoch in range(1, 2):
        loss_sum = 0.
        model.train()
        
        vloss_sum    = 0.
        vxloss_sum   = 0.
        vzloss_sum   = 0.
        vmrfloss_sum = 0.
        vqloss_sum   = 0.
        vploss_sum   = 0.
        vewcloss_sum = 0.
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                image   = val_data["image"].to(device = device)
                #_image  = model.image.hill.sample(image)
                _image  = model.image.hill.sample(image)
                vimage  = vibrate(_image) if is_vibrate else _image
                outdict = model(vimage)
                rec     = outdict["reconstruction" ]
                a       = outdict["poisson_weight" ]
                sigma   = outdict["gaussian_sigma" ]
                out     = outdict["enhanced_image" ]
                lum     = outdict["estim_luminance"]
                qloss   = outdict["quantized_loss" ]
                ploss   = outdict["psf_loss"       ]
                if adjust_luminance:
                    rec = luminance_adjustment(rec, image)
                #vloss   = loss_fn(rec, image) * loss_weight
                vloss = _loss_fnx(rec, image, a, sigma)
                vxloss_sum += vloss.detach().item()
                vloss = vloss.detach().item()             
                vloss_z = _loss_fnz(
                _input =lum ,
                mask   =None,
                target =None ) * zloss_weight
                vloss += vloss_z
                vzloss_sum += vloss_z.detach().item()
                vloss_sum += vloss.detach().item()
                if train_with_mrf:
                    loss_mrf = mrf_loss(out).detach().item()
                    vloss_sum += loss_mrf
                    vmrfloss_sum += loss_mrf
                if qloss is not None:
                    qloss = qloss.detach().item() * qloss_weight
                    vloss += qloss
                    vloss_sum += qloss
                    vqloss_sum += qloss
                if ploss is not None:
                    ploss = ploss.detach().item() * ploss_weight
                    vloss += ploss
                    vloss_sum += ploss
                    vploss_sum += ploss
                if ewc is not None:
                    ewc_loss = ewc.calc_ewc_loss(
                        ewc_weight).detach().item() * ewc_weight
                    vloss_sum += ewc_loss
                    vloss += ewc_loss
                    vewcloss_sum += ewc_loss

        num  = len(train_loader)
        vnum = len(val_loader)
        print("\nvloss_sum   \t", vloss_sum      / vnum,
              "\nvxloss_sum  \t", vxloss_sum     / vnum,
              "\nvzloss_sum  \t", vzloss_sum     / vnum,
              "\nvmrfloss_sum  \t", vmrfloss_sum / vnum,
              "\nvqloss_sum  \t", vqloss_sum     / vnum,
              "\nvploss_sum  \t", vploss_sum     / vnum,
              "\nvewcloss_sum\t", vewcloss_sum   / vnum,
              "\na    \t"       , a.detach().item(),
              "\nsigma\t"       , sigma.detach().item()
              )
        
        loss_list.append(loss_sum / num)
        vloss_list.append(vloss_sum / vnum)
        row = pd.DataFrame([[loss_list[-1], vloss_list[-1]]],
                           columns = train_curve.columns)
        train_curve = pd.concat([train_curve, row], ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv",
                           index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]},'+\
                  f' Val {vloss_list[-1]}')
        vibrate.step()
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])

def finetuning_with_simulation_loop(
        optimizer               ,
        model                   ,
        train_loader            ,
        val_loader              ,
        device                  ,
        model_name              ,
        params                  ,
        ewc                     ,
        train_dataset_params    ,
        train_loop_params       ,
        vibration_params        ,
        scheduler    = None     ,
        ):
    
    n_epochs         = train_loop_params["n_epochs"         ]
    path             = train_loop_params["path"             ]
    savefig_path     = train_loop_params["savefig_path"     ]
    adjust_luminance = train_loop_params["adjust_luminance" ]
    es_patience      = train_loop_params["es_patience"      ]
    is_vibrate       = train_loop_params["is_vibrate"       ]
    zloss_weight     = train_loop_params["zloss_weight"     ]
    ewc_weight       = train_loop_params["ewc_weight"       ]
    qloss_weight     = train_loop_params["qloss_weight"     ]
    ploss_weight     = train_loop_params["ploss_weight"     ]
    mrfloss_order    = train_loop_params["mrfloss_order"    ]
    mrfloss_weights  = train_loop_params["mrfloss_weights"  ]
    dilation         = train_loop_params["mrfloss_dilation" ]
    
    earlystopping = EarlyStopping(name        = model_name ,
                                  path        = path       ,
                                  patience    = es_patience,
                                  window_size = 3          ,
                                  metric      = "mean"     ,
                                  verbose     = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    train_curve = pd.DataFrame(columns=["training loss"    ,
                                        "validatation loss"] )
    loss_list, vloss_list= [], []
    vibrate = Vibrate(vibration_params)
    vibrate.set_arbitrary_step(100)
    mask = Mask()
    train_with_mrf = True
    mrf_loss =\
    MRFLoss(
        dims     = 3               ,
        order    = mrfloss_order   ,
        weights  = mrfloss_weights ,
        dilation = dilation        ,)
    torch.save(model.image.state_dict(),
               f"{path}/{model_name}_pre.pt")
    for epoch in range(1, n_epochs + 1):
        loss_sum = 0.
        model.train()
        for train_data in train_loader:
            torch.save(model.image.state_dict(),
                       f"{path}/{model_name}_tmp.pt")
            model.image.load_state_dict(
                torch.load(f"{path}/{model_name}_pre.pt"))
            labelz = train_data["labelz"].to(device = device)
            with torch.no_grad():
                image = imagen_instantblur(model  = model ,
                                           label  = labelz,
                                           device = device,
                                           params = params,)
                model.image.load_state_dict(
                    torch.load(f"{path}/{model_name}_tmp.pt"))
                _image  = model.image.hill.sample(image)
            _image = mask.apply_mask(
                train_dataset_params["mask"]      ,
                _image                            ,
                train_dataset_params["mask_size"] ,
                train_dataset_params["mask_num"]  ,)
            vimage = vibrate(_image) if is_vibrate else _image
            outdict = model(vimage)
            rec     = outdict["reconstruction" ]
            a       = outdict["poisson_weight" ]
            sigma   = outdict["gaussian_sigma" ]
            out     = outdict["enhanced_image" ]
            lum     = outdict["estim_luminance"]
            qloss   = outdict["quantized_loss" ]
            ploss   = outdict["psf_loss"       ]
            if adjust_luminance:
                rec = luminance_adjustment(rec, image)
            #loss  = loss_fn(rec, image) * loss_weight
            loss = _loss_fnx(rec, image, a, sigma)
            loss_z = _loss_fnz(
                _input = lum ,
                mask   = None,
                target = None ) * zloss_weight
            loss += loss_z
            if train_with_mrf:
                loss_mrf = mrf_loss(out)
                loss += loss_mrf
            if ewc is not None:
                loss += ewc.calc_ewc_loss(ewc_weight)
            if qloss is not None:
                loss += qloss * qloss_weight
            if ploss is not None:
                loss += ploss * ploss_weight
            loss_sum += loss.detach().item()
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        vloss_sum    = 0.
        vxloss_sum   = 0.
        vzloss_sum   = 0.
        vmrfloss_sum = 0.
        vqloss_sum   = 0.
        vploss_sum   = 0.
        vewcloss_sum = 0.
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                torch.save(model.image.state_dict(),
                           f"{path}/{model_name}_tmp.pt")
                model.image.load_state_dict(
                    torch.load(f"{path}/{model_name}_pre.pt"))
                labelz = val_data["labelz"].to(device = device)
                image = imagen_instantblur(model  = model ,
                                           label  = labelz,
                                           device = device,
                                           params = params,)
                _image = model.image.hill.sample(image)
                model.image.load_state_dict(
                    torch.load(f"{path}/{model_name}_tmp.pt"))
                vimage  = vibrate(_image) if is_vibrate else _image
                outdict = model(vimage)
                rec     = outdict["reconstruction" ]
                a       = outdict["poisson_weight" ]
                sigma   = outdict["gaussian_sigma" ]
                out     = outdict["enhanced_image" ]
                lum     = outdict["estim_luminance"]
                qloss   = outdict["quantized_loss" ]
                ploss   = outdict["psf_loss"       ]
                if adjust_luminance:
                    rec = luminance_adjustment(rec, image)
                #vloss   = loss_fn(rec, image) * loss_weight
                vloss = _loss_fnx(rec, image, a, sigma)
                vxloss_sum += vloss.detach().item()
                vloss = vloss.detach().item()             
                vloss_z = _loss_fnz(
                _input =lum ,
                mask   =None,
                target =None ) * zloss_weight
                vloss += vloss_z
                vzloss_sum += vloss_z.detach().item()
                vloss_sum += vloss.detach().item()
                if train_with_mrf:
                    loss_mrf = mrf_loss(out).detach().item()
                    vloss_sum += loss_mrf
                    vmrfloss_sum += loss_mrf
                if qloss is not None:
                    qloss = qloss.detach().item() * qloss_weight
                    vloss += qloss
                    vloss_sum += qloss
                    vqloss_sum += qloss
                if ploss is not None:
                    ploss = ploss.detach().item() * ploss_weight
                    vloss += ploss
                    vloss_sum += ploss
                    vploss_sum += ploss
                if ewc is not None:
                    ewc_loss = ewc.calc_ewc_loss(
                        ewc_weight).detach().item() * ewc_weight
                    vloss_sum += ewc_loss
                    vloss += ewc_loss
                    vewcloss_sum += ewc_loss

        num  = len(train_loader)
        vnum = len(val_loader)
        loss_list.append(loss_sum / num)
        vloss_list.append(vloss_sum / vnum)
        writer.add_scalar('train loss'  , loss_sum     /  num  , epoch)
        writer.add_scalar('val loss'    , vloss_sum    / vnum  , epoch)
        writer.add_scalar('val x loss'  , vxloss_sum   / vnum  , epoch)
        writer.add_scalar('val z loss'  , vzloss_sum   / vnum  , epoch)
        writer.add_scalar('val mrf loss', vmrfloss_sum / vnum  , epoch)
        writer.add_scalar('val p loss'  , vploss_sum   / vnum  , epoch)
        writer.add_scalar('val q loss'  , vqloss_sum   / vnum  , epoch)
        writer.add_scalar('val ewc loss', vewcloss_sum / vnum  , epoch)
        writer.add_scalar('a'           , a.detach().item()    , epoch)
        writer.add_scalar('sigma'       , sigma.detach().item(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)
        row = pd.DataFrame([[loss_list[-1], vloss_list[-1]]],
                           columns = train_curve.columns)
        train_curve = pd.concat([train_curve, row], ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv",
                           index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]},'+\
                  f' Val {vloss_list[-1]}')
        vibrate.step()
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])
        condition = True
        earlystopping(vloss_list[-1], model, condition = condition)
        #if earlystopping.early_stop:
        #    break
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png',
                format='png', dpi=500)

class ElasticWeightConsolidation():
    """
    modified from
    https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
    """
    def __init__(self, model, params, vibration_params, prev_dataloader, loss_fnx, loss_fnz,wx, wz,
                 init_num_batch, ewc_dataset_params, is_vibrate, device, without_noise):
        self.model = model
        num_params = 0
        for p in model.parameters():
            if p.requires_grad:
                num_params += p.numel()
        #print(num_params)
        self.num_params = num_params
        self.device = device
        self.is_vibrate = is_vibrate
        self.loss_fnx = loss_fnx
        self.loss_fnz = loss_fnz
        self.wx = wx
        self.wz = wz
        self.params = params
        self.prev_dataloader = prev_dataloader
        self.ewc_dataset_params = ewc_dataset_params
        num_fisher = 0
        self.vibrate = Vibrate(vibration_params)
        self.vibrate.set_arbitrary_step(1000)
        self.without_noise = without_noise

        for name, _ in self.model.named_buffers():
            if '_estimated_fisher' in name:
                num_fisher += 1        
        print("(ewc) registering ewc params...")
        self.register_ewc_params(prev_dataloader,
                                 num_batch=init_num_batch)

    def _update_mean_params(self):
        """
        save parameters of previous task in buffer.
        """
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean',
                                       param.data.clone())

    def _update_fisher_params(self, dataloader, num_batch=100):
        """
        calculate diagonal components of fisher information matrix on the task
        and save them in buffer.
        """
        mask = Mask()
        grad_log_likelihood_data = []
        for i, val_data in enumerate(dataloader):
            if i > num_batch:
                break
            labelx = val_data["labelx"].to(device = self.device)
            labelz = val_data["labelz"].to(device = self.device)
            with torch.no_grad():
                if self.without_noise:
                    image = imagen_instantblur_without_noise(
                        model  = self.model ,
                        label  = labelz,
                        device = self.device,
                        params = self.params,)
                else:
                    image = imagen_instantblur(
                        model  = self.model ,
                        label  = labelz     ,
                        device = self.device,
                        params = self.params,)
            image = mask.apply_mask(self.ewc_dataset_params["mask"]      ,
                                    image                                ,
                                    self.ewc_dataset_params["mask_size"] ,
                                    self.ewc_dataset_params["mask_num"]  ,)
            vimage = self.vibrate(image) if self.is_vibrate else image
            
            outdict = self.model(vimage)
            out   = outdict["enhanced_image" ]
            lum   = outdict["estim_luminance"]
            lossx  = self.loss_fnx(out,labelx)
            lossz  = _loss_fnz(
                _input = lum    ,
                mask   = labelx ,
                target = labelz  )
            vloss = self.wx * lossx + self.wz * lossz
            log_likelihood = torch.log(vloss)
            grad_log_likelihood = torch.autograd.grad(
                log_likelihood          ,
                self.model.parameters() ,
                retain_graph=False      ,
                allow_unused=True        )
            grad_log_likelihood = list(grad_log_likelihood)
            for n, param in enumerate(grad_log_likelihood):
                if param is None:
                        param_data_clone = None
                else:
                    param_data_clone = param.data.clone() / len(dataloader)
                if len(grad_log_likelihood_data) != len(grad_log_likelihood):
                    grad_log_likelihood_data.append(param_data_clone)
                elif param is not None:
                    grad_log_likelihood_data[n] += param_data_clone
        grad_log_likelihood_data = tuple(grad_log_likelihood_data)
        _buff_param_names = [param[0].replace('.', '__')
                             for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names,
                                           grad_log_likelihood_data):
            if param is None:
                param_data_clone = None
            else:
                param_data_clone = param.data.clone() ** 2 / len(dataloader)
            self.model.register_buffer(_buff_param_name+"_estimated_fisher",
                                       param_data_clone)

    def register_ewc_params(self, dataloader, num_batch):
        """
        update mean and fisher for initialization
        """
        self._update_mean_params()
        self._update_fisher_params(dataloader, num_batch)

    def calc_ewc_loss(self, lambda_):
        losses = []
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            mean = getattr(self.model, f'{_buff_param_name}_estimated_mean')
            fisher = getattr(self.model, f'{_buff_param_name}_estimated_fisher')
            if fisher is not None and mean is not None and param is not None:
                losses.append((fisher * (param - mean) ** 2).sum())
        return (lambda_ / 2) * sum(losses) / self.num_params

def loss_mode(pixelwise_loss, perceptual_loss, segmentation_loss, mode):
    if mode == "pixelwise":
        loss = pixelwise_loss + perceptual_loss * 0. + segmentation_loss * 0.
    elif mode == "perceptual":
        loss = pixelwise_loss* 0. + perceptual_loss + segmentation_loss * 0.
    elif mode == "segmentation":
        loss = pixelwise_loss* 0. +  perceptual_loss * 0. + segmentation_loss
    elif mode == "segmentation_plus_perceptual":
        loss = pixelwise_loss * 0. + segmentation_loss + perceptual_loss# referred in CellPose3
    elif mode == "pixelwise_plus_perceptual":
        loss = pixelwise_loss + perceptual_loss + segmentation_loss * 0.
    elif mode == "pixelwise_plus_segmentation":
        loss = pixelwise_loss + perceptual_loss * 0. + segmentation_loss 
    elif mode == "all_in":
        loss = pixelwise_loss + perceptual_loss + segmentation_loss
    else:
        raise ValueError(f"{mode} is not inplemented. ")
    return loss

def deep_align_net_train_loop(
        optimizer                ,
        align_model              ,
        deconv_model             ,
        train_loader             ,
        val_loader               ,
        model_name               ,
        params                   ,
        train_loop_params        ,
        train_dataset_params     ,
        vibration_params         ,
        scheduler                ,
        train_without_noise=False,
        ):
    
    n_epochs     = train_loop_params["n_epochs"]
    device       = params["device"]                               
    path         = train_loop_params["path"]            
    savefig_path = train_loop_params["savefig_path"]
    es_patience  = train_loop_params["es_patience"]
    if train_without_noise:
        loss_fn  = nn.MSELoss()
    else:
        loss_fn  = nn.GaussianNLLLoss()
    mode         = params["loss_mode"]

    earlystopping = EarlyStopping(
        name        = model_name ,
        path        = path       ,
        patience    = es_patience,
        window_size = 1          ,
        metric      = "mean"     ,
        verbose     = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    train_curve = pd.DataFrame(columns=["training loss", "validatation loss"])
    loss_list, vloss_list = [], []
    vibrate = Vibrate(vibration_params)
    vibrate.set_arbitrary_step(0)
    mask = Mask()
    for epoch in range(1, n_epochs + 1):
        loss_sum = 0.
        align_model.train()
        deconv_model.eval()
        for train_data in train_loader:
            labelz = train_data["labelz"].to(device = device)
            with torch.no_grad():
                image_with_noise = imagen_instantblur(
                    model  = deconv_model ,
                    label  = labelz       ,
                    device = device       ,
                    params = params       ,)
                image_with_noise_hill = deconv_model.image.hill.sample(image_with_noise).detach().clone()
                if train_without_noise:
                    image_without_noise = imagen_instantblur_without_noise(
                        model  = deconv_model ,
                        label  = labelz       ,
                        device = device       ,
                        params = params       ,
                    )
                    image_without_noise_hill = deconv_model.image.hill.sample(image_without_noise).detach().clone()
            vimage = vibrate(image_with_noise_hill)
            v_m_image = mask.apply_mask(
                train_dataset_params["mask"]      ,
                vimage                            ,
                train_dataset_params["mask_size"] ,
                train_dataset_params["mask_num"]  ,)
            outdict_a = align_model(v_m_image)
            aligned_image = outdict_a["aligned_image"]
            # segmentation with aligned_image 
            outdict_d = deconv_model(aligned_image)
            out_with_shake  = outdict_d["enhanced_image"]
            lum_with_shake  = outdict_d["estim_luminance"]
            mid_with_shake  = outdict_d["mid"]
            plt.imsave(f"_target_image_{model_name}.png"  , image_without_noise_hill[0, 0,:,60].detach().clone().cpu().numpy(), cmap="gray")
            plt.imsave(f"_input_image_{model_name}.png"   , vimage[0, 0,:,60].detach().clone().cpu().numpy(), cmap="gray")
            plt.imsave(f"_aligned_image{model_name}.png" , aligned_image[0, 0,:,60].detach().clone().cpu().numpy(), cmap="gray")
            # segmentation with true_image
            with torch.no_grad():
                if train_without_noise:
                    outdict_d = deconv_model(image_without_noise_hill)
                else:
                    outdict_d = deconv_model(image_with_noise_hill)
                out_without_shake  = outdict_d["enhanced_image" ]
                lum_without_shake  = outdict_d["estim_luminance"]
                mid_without_shake  = outdict_d["mid"]
            if train_without_noise:
                pixelwise_loss    = loss_fn(aligned_image, image_without_noise_hill)
                segmentation_loss = loss_fn(out_with_shake, out_without_shake)\
                                  + loss_fn(lum_with_shake, lum_without_shake)
                perceptual_loss   = loss_fn(mid_with_shake, mid_without_shake)
            else:
                pixelwise_loss    = F.gaussian_nll_loss(
                    aligned_image, image_with_noise_hill, image_with_noise_hill*0.1+0.01)
                segmentation_loss = loss_fn(out_with_shake, out_without_shake, var=torch.ones_like(out_without_shake)) \
                                  + loss_fn(lum_with_shake, lum_without_shake, var=torch.ones_like(out_without_shake))
                perceptual_loss   = loss_fn(mid_with_shake, mid_without_shake, var=torch.ones_like(mid_with_shake))

            loss = loss_mode(
                pixelwise_loss    = pixelwise_loss   ,
                perceptual_loss   = perceptual_loss  ,
                segmentation_loss = segmentation_loss,
                mode              = mode             ,
            )

            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(align_model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.detach().item()
        
        vloss_sum = 0.
        align_model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                labelz = val_data["labelz"].to(device = device)
                image_with_noise = imagen_instantblur(
                    model  = deconv_model ,
                    label  = labelz       ,
                    device = device       ,
                    params = params       ,)
                image_with_noise_hill = deconv_model.image.hill.sample(image_with_noise).detach().clone()
                if train_without_noise:
                    image_without_noise = imagen_instantblur_without_noise(
                        model  = deconv_model ,
                        label  = labelz       ,
                        device = device       ,
                        params = params       ,
                    )
                    image_without_noise_hill = deconv_model.image.hill.sample(image_without_noise).detach().clone()
                vimage = vibrate(image_with_noise_hill)
                outdict_a = align_model(vimage)
                aligned_image = outdict_a["aligned_image"]
                # segmentation with aligned_image
                outdict_d = deconv_model(aligned_image)
                out_with_shake   = outdict_d["enhanced_image" ]
                lum_with_shake   = outdict_d["estim_luminance"]
                # segmentation with true_image
                if train_without_noise:
                    outdict_d = deconv_model(image_without_noise_hill)
                else:
                    outdict_d = deconv_model(image_with_noise_hill)
                out_without_shake = outdict_d["enhanced_image" ]
                lum_without_shake = outdict_d["estim_luminance"]
                mid_without_shake = outdict_d["mid"]
                if train_without_noise:
                    pixelwise_loss    = loss_fn(aligned_image, image_without_noise_hill)
                    segmentation_loss = loss_fn(out_with_shake, out_without_shake)\
                                      + loss_fn(lum_with_shake, lum_without_shake)
                    perceptual_loss   = loss_fn(mid_with_shake, mid_without_shake)
                else:
                    pixelwise_loss    = F.gaussian_nll_loss(
                        aligned_image, image_with_noise_hill, image_with_noise_hill*0.1+0.01)
                    segmentation_loss = loss_fn(out_with_shake, out_without_shake, var=torch.ones_like(out_without_shake)) \
                                      + loss_fn(lum_with_shake, lum_without_shake, var=torch.ones_like(out_without_shake))
                    perceptual_loss   = loss_fn(mid_with_shake, mid_without_shake, var=torch.ones_like(mid_with_shake))
                vloss = loss_mode(
                    pixelwise_loss    = pixelwise_loss   ,
                    perceptual_loss   = perceptual_loss  ,
                    segmentation_loss = segmentation_loss,
                    mode              = mode             ,
                )
                vloss_sum += vloss.detach().item()
        
        num  = len(train_loader)
        vnum = len(val_loader)
        loss_list.append(loss_sum / num)
        vloss_list.append(vloss_sum / vnum)
        writer.add_scalar('train loss', loss_sum / num, epoch)
        writer.add_scalar('val loss', vloss_sum / vnum, epoch)
        row = pd.DataFrame([[loss_list[-1], vloss_list[-1]]],
                           columns = train_curve.columns)
        train_curve = pd.concat([train_curve, row], ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv",
                           index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, ' + \
                  f'Val {vloss_list[-1]}')
            
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])

        vibrate.step()
        earlystopping((vloss_sum / vnum), align_model,
                      condition = vibrate.num_step >= vibrate.max_step)
        
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    plt.legend()
    plt.savefig(
        f'{savefig_path}/{model_name}_train.png', format='png', dpi=500)
    

def finetuning_with_align_model_loop(
        optimizer              ,
        model                  ,
        align_model            ,
        train_loader           ,
        val_loader             ,
        device                 ,
        model_name             ,
        ewc                    ,
        train_dataset_params   ,
        train_loop_params      ,
        scheduler   = None     ,
        v_verbose   = False    ,
        ):
    
    n_epochs         = train_loop_params["n_epochs"         ]        
    path             = train_loop_params["path"             ]            
    savefig_path     = train_loop_params["savefig_path"     ]    
    adjust_luminance = train_loop_params["adjust_luminance" ]
    es_patience      = train_loop_params["es_patience"      ] 
    zloss_weight     = train_loop_params["zloss_weight"     ]    
    ewc_weight       = train_loop_params["ewc_weight"       ]      
    qloss_weight     = train_loop_params["qloss_weight"     ]    
    ploss_weight     = train_loop_params["ploss_weight"     ]
    mrfloss_order    = train_loop_params["mrfloss_order"    ]
    mrfloss_weights  = train_loop_params["mrfloss_weights"  ]
    dilation         = train_loop_params["mrfloss_dilation" ]
    loss_fn = nn.MSELoss()
    align_model.eval()
    
    earlystopping = EarlyStopping(
        name        = model_name ,
        path        = path       ,
        patience    = es_patience,
        window_size = 1          ,
        metric      = "mean"     ,
        verbose     = True       )
    writer = SummaryWriter(f'runs/{model_name}')
    train_curve = pd.DataFrame(columns=["training loss"    ,
                                        "validatation loss"] )
    loss_list, vloss_list= [], []
    mask = Mask()
    train_with_mrf = True
    mrf_loss =\
    MRFLoss(
        dims     = 3               ,
        order    = mrfloss_order   ,
        weights  = mrfloss_weights ,
        dilation = dilation        ,)

    for epoch in range(1, n_epochs + 1):
        loss_sum = 0.
        model.train()
        for train_data in train_loader:
            image  = train_data["image"].to(device = device)
            #_image = model.image.hill.sample(image)
            with torch.no_grad():
                _image  = model.image.hill.sample(image)
                aligned = align_model(_image)
            _image  = aligned["aligned_image"]
            image   = model.image.hill.inverse_hill(_image)
            _image = mask.apply_mask(
                train_dataset_params["mask"]      ,
                _image                            ,
                train_dataset_params["mask_size"] ,
                train_dataset_params["mask_num"]  ,)
            outdict = model(_image)
            rec     = outdict["reconstruction" ]
            a       = outdict["poisson_weight" ]
            sigma   = outdict["gaussian_sigma" ]
            lum     = outdict["estim_luminance"]
            qloss   = outdict["quantized_loss" ]
            ploss   = outdict["psf_loss"       ]
            if adjust_luminance:
                rec = luminance_adjustment(rec, image)
            #loss  = loss_fn(rec, image) * loss_weight
            loss = loss_fn(rec, image)
            loss_z = _loss_fnz(
                _input = lum ,
                mask   = None,
                target = None ) * zloss_weight
            loss += loss_z
            if ewc is not None:
                loss += ewc.calc_ewc_loss(ewc_weight)
            if qloss is not None:
                loss += qloss * qloss_weight
            if ploss is not None:
                loss += ploss * ploss_weight
            loss_sum += loss.detach().item()
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(align_model.parameters(), 1.0)
            optimizer.step()
        vloss_sum    = 0.
        vxloss_sum   = 0.
        vzloss_sum   = 0.
        vmrfloss_sum = 0.
        vqloss_sum   = 0.
        vploss_sum   = 0.
        vewcloss_sum = 0.
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                image   = val_data["image"].to(device = device)
                _image  = model.image.hill.sample(image)
                aligned = align_model(_image)
                _image  = aligned["aligned_image"]
                image   = model.image.hill.inverse_hill(_image)
                vimage  =  _image
                outdict = model(vimage)
                rec     = outdict["reconstruction" ]
                a       = outdict["poisson_weight" ]
                sigma   = outdict["gaussian_sigma" ]
                out     = outdict["enhanced_image" ]
                lum     = outdict["estim_luminance"]
                qloss   = outdict["quantized_loss" ]
                ploss   = outdict["psf_loss"       ]
                if adjust_luminance:
                    rec = luminance_adjustment(rec, image)
                #vloss   = loss_fn(rec, image) * loss_weight
                vloss = _loss_fnx(rec, image, a, sigma)
                vxloss_sum += vloss.detach().item()
                vloss = vloss.detach().item()             
                vloss_z = _loss_fnz(
                _input =lum ,
                mask   =None,
                target =None ) * zloss_weight
                vloss += vloss_z
                vzloss_sum += vloss_z.detach().item()
                vloss_sum += vloss.detach().item()
                if train_with_mrf:
                    loss_mrf = mrf_loss(out).detach().item()
                    vloss_sum += loss_mrf
                    vmrfloss_sum += loss_mrf
                if qloss is not None:
                    qloss = qloss.detach().item() * qloss_weight
                    vloss += qloss
                    vloss_sum += qloss
                    vqloss_sum += qloss
                if ploss is not None:
                    ploss = ploss.detach().item() * ploss_weight
                    vloss += ploss
                    vloss_sum += ploss
                    vploss_sum += ploss
                if ewc is not None:
                    ewc_loss = ewc.calc_ewc_loss(
                        ewc_weight).detach().item() * ewc_weight
                    vloss_sum += ewc_loss
                    vloss += ewc_loss
                    vewcloss_sum += ewc_loss

        num  = len(train_loader)
        vnum = len(val_loader)
        loss_list.append(loss_sum / num)
        vloss_list.append(vloss_sum / vnum)
        writer.add_scalar('train loss'  , loss_sum     /  num  , epoch)
        writer.add_scalar('val loss'    , vloss_sum    / vnum  , epoch)
        writer.add_scalar('val x loss'  , vxloss_sum   / vnum  , epoch)
        writer.add_scalar('val z loss'  , vzloss_sum   / vnum  , epoch)
        writer.add_scalar('val mrf loss', vmrfloss_sum / vnum  , epoch)
        writer.add_scalar('val p loss'  , vploss_sum   / vnum  , epoch)
        writer.add_scalar('val q loss'  , vqloss_sum   / vnum  , epoch)
        writer.add_scalar('val ewc loss', vewcloss_sum / vnum  , epoch)
        writer.add_scalar('a'           , a.detach().item()    , epoch)
        writer.add_scalar('sigma'       , sigma.detach().item(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)
        row = pd.DataFrame([[loss_list[-1], vloss_list[-1]]],
                           columns = train_curve.columns)
        train_curve = pd.concat([train_curve, row], ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv",
                           index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]},'+\
                  f' Val {vloss_list[-1]}')
        
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])
        earlystopping(vloss_list[-1], model, condition=True)
        #if earlystopping.early_stop:
        #    break
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png',
                format='png', dpi=500)