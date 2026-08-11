
# dinh nghia kien truc model
def generator(args):
    dropout = args.dropout
    if args.encoder =='b_tcn':
        pass
    elif args.encoder=='trans':
        pass



    if args.mode == 'train':




    if opt.mode == 'test': dropout = 0.


    net = TCN.Local_TCNNet(num_inputs=opt.dimension,
                           num_channels=opt.num_channels,
                           dropout=dropout,
                           activation=opt.activation,
                           window_size=opt.w,
                           predicted_length=opt.predicted_length,
                           )
    if opt.mode =='train':
        training_model = net
        return training_model
    elif opt.mode =='test':
        testting_model = net
        return testting_model

    pass