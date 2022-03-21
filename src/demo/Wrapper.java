package demo;

public class Wrapper {
    private String netName;
    private String activator;
    private String optimizer;
    private String lossFunc;
    private String[] opParams;
    private String[] lossParams;
    private int level;
    private int epoch;
    private int[] inDi;
    private int[] outDi;

    public String getNetName() {
        return netName;
    }

    public void setNetName(String netName) {
        this.netName = netName;
    }

    public String getActivator() {
        return activator;
    }

    public void setActivator(String activator) {
        this.activator = activator;
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
}
