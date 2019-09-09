
def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'PATN':
        assert opt.dataset_mode == 'keypoint'
        from .PATN import TransferModel
        model = TransferModel()
    elif opt.model == 'PATN_Deform':
        assert opt.dataset_mode == 'key_segments'
        from .PATN_Deform import TransferModel
        model = TransferModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
