package demo;

import java.util.HashMap;
import java.util.Map;

public class Wrapper {
    private String nodeId;
    private String netName;
    private String[] activators;
    private String optimizer;
    private String lossFunc;
    private String[] opParams;
    private String[] lossParams;
    private int level;
    private int epoch;
    private int[] inDi;
    private int[] outDi;

    public String getNodeId() {
        return nodeId;
    }

    public void setNodeId(String nodeId) {
        this.nodeId = nodeId;
    }

    public String getNetName() {
        return netName;
    }

    public void setNetName(String netName) {
        this.netName = netName;
    }

    public String[] getActivators() {
        return activators;
    }

    public void setActivators(String[] activators) {
        this.activators = activators;
    }

    public String getOptimizer() {
        return optimizer;
    }

    public void setOptimizer(String optimizer) {
        this.optimizer = optimizer;
    }

    public String getLossFunc() {
        return lossFunc;
    }

    public void setLossFunc(String lossFunc) {
        this.lossFunc = lossFunc;
    }

    public String[] getOpParams() {
        return opParams;
    }

    public void setOpParams(String[] opParams) {
        this.opParams = opParams;
    }

    public String[] getLossParams() {
        return lossParams;
    }

    public void setLossParams(String[] lossParams) {
        this.lossParams = lossParams;
    }

    public int getLevel() {
        return level;
    }

    public void setLevel(int level) {
        this.level = level;
    }

    public int getEpoch() {
        return epoch;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    public int[] getInDi() {
        return inDi;
    }

    public void setInDi(int[] inDi) {
        this.inDi = inDi;
    }

    public int[] getOutDi() {
        return outDi;
    }

    public void setOutDi(int[] outDi) {
        this.outDi = outDi;
    }

    public String formatActivator() {
        StringBuilder builder = new StringBuilder("");
        for (String s : this.getActivators()) {
            builder.append(s + " ");
        }
        return builder.toString().trim();
    }

    public String formatInDi() {
        StringBuilder builder = new StringBuilder("");
        for (int i : this.getInDi()) {
            builder.append(i + " ");
        }
        return builder.toString().trim();
    }

    public String formatOutDi() {
        StringBuilder builder = new StringBuilder("");
        for (int i : this.getOutDi()) {
            builder.append(i + " ");
        }
        return builder.toString().trim();
    }

//# optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False) 5
//# optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0) 5
//# optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)) 3
//# optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0) 3
//# optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0) 4
//# optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False) 6
//# optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 5
//# optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 4

    public String formatOp() {
        StringBuilder builder = new StringBuilder("");
        //
        Map<String, String[]> map = new HashMap<>();
        map.put("SGD", new String[]{"momentum=", "dampening=", "weight_decay=", "neserov="});
        map.put("ASGD", new String[]{"lambd=", "alpha=", "t0=", "weight_decay="});
        map.put("Rprop", new String[]{"etas=", "step_sizes="});
        map.put("Adagrad", new String[]{"lr_decay=", "weight_decay="});
        map.put("Adadelta", new String[]{"rho=", "eps=", "weight_decay="});
        map.put("RMSprop", new String[]{"alpha=", "eps=", "weight_decay=", "momentum=", "centered="});
        map.put("Adam", new String[]{"betas=", "eps=", "weight_decay="});
        map.put("Adamax", new String[]{"betas=", "eps=", "weight_decay="});
        builder.append("#" + this.getOptimizer() + "(params, lr=");
        builder.append(this.getOpParams()[0] + ", ");
        String[] texts = map.get(this.getOptimizer());
        for (int i = 0; i < texts.length; i++) {
            builder.append(texts[i] + this.getOpParams()[i + 1]);
            if (i != texts.length - 1) builder.append(", ");
        }
        builder.append(")\n");
        return builder.toString();
    }

//torch.nn.BCELoss(weight=None, size_average=True, reduce=True, reduction='mean') 4
//torch.nn.BCEWithLogitsLoss(weight=None, size_average=True, reduce=True, reduction='mean', pos_weight=None) 5
//torch.nn.NLLLoss(weight=None, size_average=None, reduce=None, reduction='mean') 4
//torch.nn.CrossEntropyLoss(weight=None, size_average=True) 2
//torch.nn.L1Loss(size_average=True, reduce=True, reduction='mean') 3
//torch.nn.MSELoss(size_average=True, reduce=True, reduction='mean') 3
//torch.nn.SmoothL1Loss(size_average=True, reduce=True, reduction='mean', beta=1.0) 4

    public String formatLoss() {
        StringBuilder builder = new StringBuilder("");
        //
        Map<String, String[]> map = new HashMap<>();
        map.put("BCELoss", new String[]{"weight=", "size_average=", "reduce=", "reduction="});
        map.put("BCEWithLogitsLoss", new String[]{"weight=", "size_average=", "reduce=", "reduction=", "pos_weight="});
        map.put("NLLLoss", new String[]{"weight=", "size_average=", "reduce=", "reduction="});
        map.put("CrossEntropyLoss", new String[]{"weight=", "size_average="});
        map.put("L1Loss", new String[]{"size_average=", "reduce=", "reduction="});
        map.put("MSELoss", new String[]{"size_average=", "reduce=", "reduction="});
        map.put("SmoothL1Loss", new String[]{"size_average=", "reduce=", "reduction=", "beta="});
        builder.append("#" + this.getLossFunc() + "(");
        String[] texts = map.get(this.getLossFunc());
        for (int i = 0; i < texts.length; i++) {
            builder.append(texts[i] + this.getLossParams()[i]);
            if (i != texts.length - 1) builder.append(", ");
        }
        builder.append(")\n");
        return builder.toString();
    }

    public String toText() {
        StringBuilder builder = new StringBuilder("");
        builder.append("#" + this.getNodeId() + "\n");
        builder.append("#" + this.getNetName() + "\n");
        builder.append("#" + this.getLevel() + "\n");
        builder.append("#" + this.formatActivator() + "\n");
        builder.append(formatOp());
        builder.append(formatLoss());
        builder.append("#" + this.getEpoch() + "\n");
        builder.append("#" + this.formatInDi() + "\n");
        builder.append("#" + this.formatOutDi() + "\n");
        return builder.toString();
    }
}
