# Combine L/R masks into one mask...
# Add scratch code here

# roi_ref = {"CTVn":["LCTVn", "RCTVn"], "Glnd_Lacrimal":["Glnd_Lacrimal_L","Glnd_Lacrimal_R"],
#            "Spc_Retrophar":["Spc_Retrophar_R", "Spc_Retrophar_L"],
#            "Lens":["Lens_L", "Lens_R"], "Eye":["Eye_L", "Eye_R"],
#            "Nrv_Optic":["Nrv_Optic_L", "Nrv_Optic_R"], "Parotid":["Parotid_L", "Parotid_R"],
#            "Cochlea":["Cochlea_L", "Cochlea_R"],"Glnd_Submand":["Glnd_Submand_L",
#            "Glnd_Submand_R"], "BrachialPlex":["BrachialPlex_R", "BrachialPlex_L"],}

# ROIS = [ "Brain", "Lips", "Brainstem", "Esophagus", "Larynx", "OpticChiasm", "SpinalCord", 
#          "Mandible_Bone", "CTVn", "Glnd_Lacrimal","Spc_Retrophar", "Lens", "Eye","Nrv_Optic", 
#          "Parotid", "Cochlea", "Glnd_Submand", "BrachialPlex"]

# "Musc_Constrict":["Musc_Constrict_I", "Musc_Constrict_S", "Musc_Constrict_M"]}
# "External",    "LEVEL_IA"
# roi_ref= {"LEVEL_IB":["LEVEL_IB_RT","LEVEL_IB_LT"], "LEVEL_III":["LEVEL_III_RT","LEVEL_III_LT"], "LEVEL_II":["LEVEL_II_RT","LEVEL_II_LT"], 
#           "LEVEL_IV":["LEVEL_IV_RT", "LEVEL_IV_LT"], "LEVEL_V":["LEVEL_V_RT","LEVEL_V_LT"], "LEVEL_VIIA":["LEVEL_VIIA_RT", "LEVEL_VIIA_LT"]}

# import random
# new_ids = []
# oars = []
# folders = glob.glob("*")
# random.shuffle(folders)
# for i, fo in enumerate(folders):
#     fold = glob.glob(fo+"/structures/*")
#     fold_ = [f.split("/")[-1].partition(".")[0] for f in fold]
    # after combining L/R sides...
    # oars += fold_
    # new_ids += [fo for i in range(len(fold_))]
    # for c in list(roi_ref.keys()):
    #     vals = roi_ref[c]
    #     count = 0
    #     for v in vals:
    #         if v in fold_:
    #             count +=1
    #     if count == 2:
    #         if os.path.isfile(f+f"/structures/{c}.nrrd") is False:
    #             paths = [f+f"/structures/{v}.nrrd" for v in vals]
    #             mask = nrrd.read(paths[0])
    #             header = mask[1]
    #             mask = mask[0]
    #             mask_ = nrrd.read(paths[1])
    #             mask += mask_[0]
    #             header["ROI"] = c
    #             header["Voxels"] = len(mask[mask==1])
    #             nrrd.write(f+f"/structures/{c}.nrrd", mask, header=header)
    #             print(f"Saved {c} for {f}. {i}")
    

# fold = glob.glob(f+"/structures/*")
# fold_ = [f.split("/")[-1].partition(".")[0] for f in fold]
# ids_ = [f for i in range()]


#             # break
#     # break

# ROIS_REMAIN = ["External", "GTVp", "BRAIN", "OralCavity", "Lips", "Brainstem", "Esophagus",
#         "Larynx", "Cricoid_P", "OpticChiasm", "SpinalCord", "Mandible_Bone", 
#         "LEVEL_IA", "LEVEL_IB_RT", "LEVEL_III_RT", "LEVEL_II_RT", "LEVEL_IV_RT", "LEVEL_VIIA_RT", "LEVEL_V_RT",
#         "LEVEL_IB_LT", "LEVEL_III_LT", "LEVEL_II_LT", "LEVEL_IV_LT", "LEVEL_VIIA_LT", "LEVEL_V_LT"]

# naming = { "Brainstem":["BRAIN_STEM"], "OpticChiasm":["CHIASM"], "Lens_L":["L_LENS", "LT_LENS"], "Lens_R": ["R_LENS", "RT_LENS"], "Eye_L": ["L_EYE", "LT_EYE"], "Eye_R": ["R_EYE", "RT_EYE"],
#     "Nrv_Optic": ["OPTICS"], "Nrv_Optic_L": ["L_OPTIC", "LT_OPTIC"], "Nrv_Optic_R": ["R_OPTIC", "RT_OPTIC"], "Parotid_L": ["LT_PAROTID", "L_PAROTID"], "Parotid_R": ["R_PAROTID", "RT_PAROTID"],
#     "Lung_L": ["L_LUNG", "LT_LUNG"], "Lung_R": ["R_LUNG", "RT_LUNG"], "Brain":["BRAIN"]}

# BRAIN_STEM.nrrd  External.nrrd  LCTVn.nrrd      LPTV56.nrrd     OPT_LPTV56.nrrd  OS_LT_PARO.nrrd    PTV70.nrrd   R_PAROTID.nrrd  SpinalCord.nrrd
# CARTILAGE.nrrd   EXT_VOLS.nrrd  L_LUNG.nrrd     LT_LUNG.nrrd    OPT_PTV70.nrrd   OS_RT_PARO.nrrd    RCTVn.nrrd   RPTV56.nrrd
# CTV70.nrrd       GTVp.nrrd      L_PAROTID.nrrd  OPT_CTV70.nrrd  OPT_RPTV56.nrrd  POST_EXT_VOL.nrrd  R_LUNG.nrrd  RT_LUNG.nrrd

# rename_ = []
# for f in folders:
#     fold = glob.glob(f+"/structures/*")
#     for c in fold:
#         if "EXTERNAL" not in c.upper():
#             oar = c.split("/")[-1]
#             if "RT_" in oar[:4]:
#                 rename_.append(f)
#                 break
#             elif "R_" in oar[:4]:
#                 rename_.append(f)
#                 break
            
# for re_ in folders:
#     fold = glob.glob(re_+"/structures/*")
#     for o in oars:
#         compare = naming[o]
#         for c in fold:
#             oar = c.split("/")[-1]#.partition(".")[0]
#             if ".nrrd" not in oar:
#                 print(oar)
#                 try:
#                     shutil.move(c, c+".nrrd")
#                 except Exception as e:
#                     print(str(e))
#             # if oar in compare:
#             #     shutil.move(c, re_+"/structures/"+o)
#             #     print(oar+" moved to "+re_+"/structures/"+o)
#     print("Done with "+re_)
#     # break