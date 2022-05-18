package demo;


import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class Controller {
    private Stage stage;
    public FileChooser fileChooser = new FileChooser();
    private String outputString = "../output";
//    private String outputString = "../../output";
    private String currentPath;
    private String pngPath;
    private String logName;
    private String logAbsoluteDir;
    private Wrapper wrapper;
    private Map<String, Integer> opMap = new HashMap<>();
    private Map<String, Integer> lossMap = new HashMap<>();
    private Map<Integer, String> nodeMap = new HashMap<>();


    @FXML
    public HBox outBox;
    public AnchorPane ap;
    public Button readbt;
    public TextField nodeIdField;
    public ComboBox<String> netcb;
    public ComboBox<String> levelcb;
    public AnchorPane activatorPane;
    public ComboBox<String> actcb1;
    public ComboBox<String> actcb2;
    public ComboBox<String> actcb3;
    public ComboBox<String> actcb4;
    public ComboBox<String> actcb5;
    public ComboBox<String> actcb6;
    public ComboBox<String> actcb7;
    public ComboBox<String> actcb8;
    public ComboBox<String> opcb;
    public AnchorPane opPane;
    public Text opText1;
    public Text opText2;
    public Text opText3;
    public Text opText4;
    public Text opText5;
    public Text opText6;
    public TextField opParam1;
    public TextField opParam2;
    public TextField opParam3;
    public TextField opParam4;
    public TextField opParam5;
    public TextField opParam6;
    public ComboBox<String> losscb;
    public AnchorPane lossPane;
    public Text lossText1;
    public Text lossText2;
    public Text lossText3;
    public Text lossText4;
    public Text lossText5;
    public TextField lossParam1;
    public TextField lossParam2;
    public TextField lossParam3;
    public TextField lossParam4;
    public TextField lossParam5;
    public AnchorPane lastPane;
    public TextField epochField;
    public TextField inDiField;
    public TextField outDiField;
    public Button saveSetting;
    public Button generate;
    public TextArea infoArea;

    public Tab imageTab;
    public ImageView imageView;
    public Tab reportTab;
    public Button trainbt;
    public Button testbt;
    public TextArea reportArea;

    /**
     * 所有文件目录不能带有空格！！！
     */

    @FXML
    public void readFile() {
        fileChooser.setTitle("open log file");
        File file = fileChooser.showOpenDialog(stage);
        String fileName = file.getName();   //fileName带文件后缀
        // 输入日志的绝对路径
        String logPath = file.getAbsolutePath();
        String[] strs = fileName.split("\\.");
        this.logName = strs[0];
        String logDir = outputString + "/" + logName;
        File logDirFile = new File(logDir);
        if (!logDirFile.exists()) {
            logDirFile.mkdir();
        }
        // copy日志文件到output/log1/log1.txt目录下
        File targetLog = new File(logDir + "/" + fileName);
        if (!targetLog.exists()) {
            try {
                Files.copy(Paths.get(logPath), Paths.get(logDir + "/" + fileName));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        LocalTime time = LocalTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        String infoText = infoArea.getText();
        String currentTime = time.format(formatter);
        infoText += currentTime + " 成功读取日志" + logName + "\n";
        infoArea.setText(infoText);
        // output/log1/ output目录下的绝对路径
        this.logAbsoluteDir = targetLog.getAbsolutePath().replace(fileName, "");
        this.currentPath = System.getProperty("user.dir");
//        System.out.println(currentPath);
        Process process;
        // 读取日志，生成json
        // java -jar ProgramCallLogic_Code.jar C:\H\Java_codes\JavaFX-GUI-demo\..\output\solution12_small\
        // C:\H\Java_codes\JavaFX-GUI-demo\..\output\solution12_small\
        String jsonPath = logAbsoluteDir + "data1.json";
        if (!new File(jsonPath).exists()) {
            try {
                String cmdstr = "java -jar ProgramCallLogic_Code.jar " + logAbsoluteDir + " " + logAbsoluteDir;
//              System.out.println(logAbsoluteDir);
//              System.out.println(currentPath);
                process = Runtime.getRuntime().exec(cmdstr, null, new File(this.currentPath));
                process.waitFor();
            } catch (Throwable e) {
                e.printStackTrace();
            }
        }
        time = LocalTime.now();
        currentTime = time.format(formatter);
        infoText = infoArea.getText();
        infoText += currentTime + " 由日志生成json文件\n";
        infoArea.setText(infoText);
        // 从整棵流程树的txt生成每一个节点的txt
        String txtPath = logAbsoluteDir + fileName;
        try {
            String cmdstr = "python log2data.py " + txtPath;
            process = Runtime.getRuntime().exec(cmdstr, null, new File(currentPath + "\\model"));
            process.waitFor();
        } catch (Throwable e) {
            e.printStackTrace();
        }
        // json生成流程树png图像
        String dotPath = logAbsoluteDir + "diagram.dot";
        this.pngPath = logAbsoluteDir + "tree.png";
        if (!new File(dotPath).exists()) {
            try {
                String pyDir = currentPath + "\\model";
                String cmdstr = "python json2diagram.py " + jsonPath + " " + logAbsoluteDir;
                process = Runtime.getRuntime().exec(cmdstr, null, new File(pyDir));
                process.waitFor();
            } catch (Throwable e) {
                e.printStackTrace();
            }
        }
        time = LocalTime.now();
        currentTime = time.format(formatter);
        infoText = infoArea.getText();
        infoText += currentTime + " 成功生成流程树图像，点击流程树图像tab查看\n";
        infoArea.setText(infoText);
        // 从dot文件获得节点id和它对应的节点
        // 保存在map
        this.nodeMap = getIdMap(dotPath);
    }

    @FXML
    public void selectNet() {
        String netName = netcb.getValue();
        wrapper.setNetName(netName);
    }

    @FXML
    public void selectLevel() {
        String levelstr = levelcb.getValue();
        int level = Integer.parseInt(levelstr);
        wrapper.setLevel(level);
        activatorPane.setVisible(true);
        switch (level) {
            case 1:
                actcb1.setVisible(true);
                actcb2.setVisible(false);
                actcb3.setVisible(false);
                actcb4.setVisible(false);
                actcb5.setVisible(false);
                actcb6.setVisible(false);
                actcb7.setVisible(false);
                actcb8.setVisible(false);
                break;
            case 2:
                actcb1.setVisible(true);
                actcb2.setVisible(true);
                actcb3.setVisible(false);
                actcb4.setVisible(false);
                actcb5.setVisible(false);
                actcb6.setVisible(false);
                actcb7.setVisible(false);
                actcb8.setVisible(false);
                break;
            case 3:
                actcb1.setVisible(true);
                actcb2.setVisible(true);
                actcb3.setVisible(true);
                actcb4.setVisible(false);
                actcb5.setVisible(false);
                actcb6.setVisible(false);
                actcb7.setVisible(false);
                actcb8.setVisible(false);
                break;
            case 4:
                actcb1.setVisible(true);
                actcb2.setVisible(true);
                actcb3.setVisible(true);
                actcb4.setVisible(true);
                actcb5.setVisible(false);
                actcb6.setVisible(false);
                actcb7.setVisible(false);
                actcb8.setVisible(false);
                break;
            case 5:
                actcb1.setVisible(true);
                actcb2.setVisible(true);
                actcb3.setVisible(true);
                actcb4.setVisible(true);
                actcb5.setVisible(true);
                actcb6.setVisible(false);
                actcb7.setVisible(false);
                actcb8.setVisible(false);
                break;
            case 6:
                actcb1.setVisible(true);
                actcb2.setVisible(true);
                actcb3.setVisible(true);
                actcb4.setVisible(true);
                actcb5.setVisible(true);
                actcb6.setVisible(true);
                actcb7.setVisible(false);
                actcb8.setVisible(false);
                break;
            case 7:
                actcb1.setVisible(true);
                actcb2.setVisible(true);
                actcb3.setVisible(true);
                actcb4.setVisible(true);
                actcb5.setVisible(true);
                actcb6.setVisible(true);
                actcb7.setVisible(true);
                actcb8.setVisible(false);
                break;
            case 8:
                actcb1.setVisible(true);
                actcb2.setVisible(true);
                actcb3.setVisible(true);
                actcb4.setVisible(true);
                actcb5.setVisible(true);
                actcb6.setVisible(true);
                actcb7.setVisible(true);
                actcb8.setVisible(true);
        }
    }
//relu
//sigmoid
//tanh
//softplus


    @FXML
    public void selectOp() {
        String op = opcb.getValue();
        opPane.setVisible(true);
        opText1.setText("lr:");
        opText1.setVisible(true);
        opParam1.setVisible(true);
        opText2.setVisible(true);
        opParam2.setVisible(true);
        opText3.setVisible(true);
        opParam3.setVisible(true);
        switch (op) {
            case "SGD":
                opText2.setText("momentum:");
                opParam2.setText("0");
                opText3.setText("dampening:");
                opParam3.setText("0");
                opText4.setText("weight_decay:");
                opText4.setVisible(true);
                opParam4.setText("0");
                opParam4.setVisible(true);
                opText5.setText("nesterov:");
                opText5.setVisible(true);
                opParam5.setText("False");
                opParam5.setVisible(true);
                opText6.setVisible(false);
                opParam6.setVisible(false);
                break;
            case "ASGD":
                opText2.setText("lambd:");
                opParam2.setText("0.0001");
                opText3.setText("alpha:");
                opParam3.setText("0.75");
                opText4.setText("t0:");
                opText4.setVisible(true);
                opParam4.setText("1000000.0");
                opParam4.setVisible(true);
                opText5.setText("weight_decay:");
                opText5.setVisible(true);
                opParam5.setText("0");
                opParam5.setVisible(true);
                opText6.setVisible(false);
                opParam6.setVisible(false);
                break;
            case "Adam":
            case "Adamax":
                opText2.setText("betas:");
                opParam2.setText("(0.9, 0.999)");
                opText3.setText("eps:");
                opParam3.setText("0.00000001");
                opText4.setText("weight_decay:");
                opText4.setVisible(true);
                opParam4.setText("0");
                opParam4.setVisible(true);
                opText5.setVisible(false);
                opParam5.setVisible(false);
                opText6.setVisible(false);
                opParam6.setVisible(false);
                break;
            case "Adadelta":
                opText2.setText("rho:");
                opParam2.setText("0.9");
                opText3.setText("eps:");
                opParam3.setText("0.000001");
                opText4.setText("weight_decay:");
                opText4.setVisible(true);
                opParam4.setText("0");
                opParam4.setVisible(true);
                opText5.setVisible(false);
                opParam5.setVisible(false);
                opText6.setVisible(false);
                opParam6.setVisible(false);
                break;
            case "Adagrad":
                opText2.setText("lr_decay:");
                opParam2.setText("0");
                opText3.setText("weight_decay:");
                opParam3.setText("0");
                opText4.setVisible(false);
                opParam4.setVisible(false);
                opText5.setVisible(false);
                opParam5.setVisible(false);
                opText6.setVisible(false);
                opParam6.setVisible(false);
                break;
            case "Rprop":
                opText2.setText("etas:");
                opParam2.setText("(0.5, 1.2)");
                opText3.setText("step_sizes:");
                opParam3.setText("(0.000001, 50)");
                opText4.setVisible(false);
                opParam4.setVisible(false);
                opText5.setVisible(false);
                opParam5.setVisible(false);
                opText6.setVisible(false);
                opParam6.setVisible(false);
                break;
            case "RMSprop":
                opText2.setText("alpha:");
                opParam2.setText("0.99");
                opText3.setText("eps:");
                opParam3.setText("0.00000001");
                opText4.setText("weight_decay:");
                opText4.setVisible(true);
                opParam4.setText("0");
                opParam4.setVisible(true);
                opText5.setText("momentum:");
                opText5.setVisible(true);
                opParam5.setText("0");
                opParam5.setVisible(true);
                opText6.setText("centered:");
                opText6.setVisible(true);
                opParam6.setText("False");
                opParam6.setVisible(true);
                break;
        }
        wrapper.setOptimizer(op);
    }

//# optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False) 5
//# optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0) 5
//# optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)) 3
//# optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0) 3
//# optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0) 4
//# optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False) 6
//# optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 5
//# optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 4

    @FXML
    public void selectLoss() {
        String loss = losscb.getValue();
        lossPane.setVisible(true);
        switch (loss) {
            case "BCELoss":
                lossText1.setText("weight:");
                lossText1.setVisible(true);
                lossParam1.setText("None");
                lossParam1.setVisible(true);
                lossText2.setText("size_average:");
                lossText2.setVisible(true);
                lossParam2.setText("True");
                lossParam2.setVisible(true);
                lossText3.setText("reduce:");
                lossText3.setVisible(true);
                lossParam3.setText("True");
                lossParam3.setVisible(true);
                lossText4.setText("reduction:");
                lossText4.setVisible(true);
                lossParam4.setText("'mean'");
                lossParam4.setVisible(true);
                lossText5.setVisible(false);
                lossParam5.setVisible(false);
                break;
            case "NLLLoss":
                lossText1.setText("weight:");
                lossText1.setVisible(true);
                lossParam1.setText("None");
                lossParam1.setVisible(true);
                lossText2.setText("size_average:");
                lossText2.setVisible(true);
                lossParam2.setText("None");
                lossParam2.setVisible(true);
                lossText3.setText("reduce:");
                lossText3.setVisible(true);
                lossParam3.setText("None");
                lossParam3.setVisible(true);
                lossText4.setText("reduction:");
                lossText4.setVisible(true);
                lossParam4.setText("'mean'");
                lossParam4.setVisible(true);
                lossText5.setVisible(false);
                lossParam5.setVisible(false);
                break;
            case "BCEWithLogitsLoss":
                lossText1.setText("weight:");
                lossText1.setVisible(true);
                lossParam1.setText("None");
                lossParam1.setVisible(true);
                lossText2.setText("size_average:");
                lossText2.setVisible(true);
                lossParam2.setText("True");
                lossParam2.setVisible(true);
                lossText3.setText("reduce:");
                lossText3.setVisible(true);
                lossParam3.setText("True");
                lossParam3.setVisible(true);
                lossText4.setText("reduction:");
                lossText4.setVisible(true);
                lossParam4.setText("'mean'");
                lossParam4.setVisible(true);
                lossText5.setText("pos_weight:");
                lossText5.setVisible(true);
                lossParam5.setText("None");
                lossParam5.setVisible(true);
                break;
            case "L1Loss":
            case "MSELoss":
                lossText1.setText("size_average:");
                lossText1.setVisible(true);
                lossParam1.setText("True");
                lossParam1.setVisible(true);
                lossText2.setText("reduce:");
                lossText2.setVisible(true);
                lossParam2.setText("True");
                lossParam2.setVisible(true);
                lossText3.setText("reduction:");
                lossText3.setVisible(true);
                lossParam3.setText("'mean'");
                lossParam3.setVisible(true);
                lossText4.setVisible(false);
                lossParam4.setVisible(false);
                lossText5.setVisible(false);
                lossParam5.setVisible(false);
                break;
            case "CrossEntropyLoss":
                lossText1.setText("weight:");
                lossText1.setVisible(true);
                lossParam1.setText("None");
                lossParam1.setVisible(true);
                lossText2.setText("size_average:");
                lossText2.setVisible(true);
                lossParam2.setText("True");
                lossParam2.setVisible(true);
                lossText3.setVisible(false);
                lossParam3.setVisible(false);
                lossText4.setVisible(false);
                lossParam4.setVisible(false);
                lossText5.setVisible(false);
                lossParam5.setVisible(false);
                break;
            case "SmoothL1Loss":
                lossText1.setText("size_average:");
                lossText1.setVisible(true);
                lossParam1.setText("True");
                lossParam1.setVisible(true);
                lossText2.setText("reduce:");
                lossText2.setVisible(true);
                lossParam2.setText("True");
                lossParam2.setVisible(true);
                lossText3.setText("reduction:");
                lossText3.setVisible(true);
                lossParam3.setText("'mean'");
                lossParam3.setVisible(true);
                lossText4.setText("beta:");
                lossText4.setVisible(true);
                lossParam4.setText("1.0");
                lossParam4.setVisible(true);
                lossText5.setVisible(false);
                lossParam5.setVisible(false);
                break;
        }
        wrapper.setLossFunc(loss);
        lastPane.setVisible(true);
    }
//torch.nn.BCELoss(weight=None, size_average=True, reduce=True, reduction='mean') 4
//torch.nn.BCEWithLogitsLoss(weight=None, size_average=True, reduce=True, reduction='mean', pos_weight=None) 5
//torch.nn.NLLLoss(weight=None, size_average=None, reduce=None, reduction='mean') 4
//torch.nn.CrossEntropyLoss(weight=None, size_average=True) 2
//torch.nn.L1Loss(size_average=True, reduce=True, reduction='mean') 3
//torch.nn.MSELoss(size_average=True, reduce=True, reduction='mean') 3
//torch.nn.SmoothL1Loss(size_average=True, reduce=True, reduction='mean', beta=1.0) 4


    @FXML
    public void saveAll() {
        Alert saveAlert = new Alert(
                Alert.AlertType.CONFIRMATION,
                "保存参数后不可修改，是否确认保存？",
                ButtonType.YES,
                ButtonType.NO,
                ButtonType.CANCEL
        );
        saveAlert.setTitle("Confirm");
        saveAlert.showAndWait();
        if (saveAlert.getResult() == ButtonType.YES) {
            //
            disableAll();
//            saveFirst();
            wrapper.setNodeId(nodeIdField.getText());
            saveActivators();
            saveOp();
            saveLoss();
            saveLast();
            if (wrapper.allPrepared())
                generate.setDisable(false);

            LocalTime time = LocalTime.now();
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");
            String infoText = infoArea.getText();
            String currentTime = time.format(formatter);
            infoText += currentTime + " 成功保存参数设置\n";
            infoArea.setText(infoText);
        }
    }

    private void saveActivators() {
        int num = wrapper.getLevel();
        String[] activators = new String[num];
        List<String> list = new ArrayList<>();
        list.add(actcb1.getValue());
        list.add(actcb2.getValue());
        list.add(actcb3.getValue());
        list.add(actcb4.getValue());
        list.add(actcb5.getValue());
        list.add(actcb6.getValue());
        list.add(actcb7.getValue());
        list.add(actcb8.getValue());
        for (int i = 0; i < num; i++)
            activators[i] = list.get(i);
        wrapper.setActivators(activators);
    }


    public void saveOp() {
        String op = wrapper.getOptimizer();
        int num = opMap.get(op);
        String[] opParams = new String[num];
        opParams[0] = opParam1.getText();
        opParams[1] = opParam2.getText();
        opParams[2] = opParam3.getText();
        switch (op) {
            case "SGD":
            case "ASGD":
            case "Adam":
                opParams[3] = opParam4.getText();
                opParams[4] = opParam5.getText();
                break;
            case "Adadelta":
            case "Adamax":
                opParams[3] = opParam4.getText();
                break;
            case "RMSprop":
                opParams[3] = opParam4.getText();
                opParams[4] = opParam5.getText();
                opParams[5] = opParam5.getText();
                break;
            case "Rprop":
            case "Adagrad":
                break;
        }
        wrapper.setOpParams(opParams);
    }

    public void saveLoss() {
        String loss = wrapper.getLossFunc();
        int num = lossMap.get(loss);
        String[] lossParams = new String[num];
        lossParams[0] = lossParam1.getText();
        lossParams[1] = lossParam2.getText();
        switch (loss) {
            case "BCELoss":
            case "NLLLoss":
            case "SmoothL1Loss":
                lossParams[2] = lossParam3.getText();
                lossParams[3] = lossParam4.getText();
                break;
            case "BCEWithLogitsLoss":
                lossParams[2] = lossParam3.getText();
                lossParams[3] = lossParam4.getText();
                lossParams[4] = lossParam5.getText();
                break;
            case "L1Loss":
            case "MSELoss":
                lossParams[2] = lossParam3.getText();
                break;
            case "CrossEntropyLoss":
                break;

        }
        wrapper.setLossParams(lossParams);
    }

    public void saveLast() {
        String epochstr = epochField.getText();
        int e = isInt(epochstr);
        if (e > 0) wrapper.setEpoch(e);
        String[] inDimensions = inDiField.getText().split(",");
        String[] outDimensions = outDiField.getText().split(",");
        List<Integer> list = new ArrayList<>();
        for (String s : inDimensions) {
            int res = isInt(s);

            if (res > Integer.MIN_VALUE) list.add(res);
        }
        int[] inArray = new int[wrapper.getLevel()];
        for (int i = 0; i < list.size(); i++) inArray[i] = list.get(i);
        wrapper.setInDi(inArray);
//        for (int i : tmpArray) System.out.println(i);
        //
        list.clear();
        int[] outArray = new int[wrapper.getLevel()];
        for (String s : outDimensions) {
            int res = isInt(s);
            if (res > Integer.MIN_VALUE) list.add(res);
        }
        for (int i = 0; i < list.size(); i++) outArray[i] = list.get(i);
        wrapper.setOutDi(outArray);
//        for (int i : tmpArray) System.out.println(i);
        //
    }

    private void checkInput(String input) {
        String tmp = input.replace(" ", "");

    }

    private int isInt(String str) {
        try {
            int res = Integer.parseInt(str);
            return res;
        } catch (NumberFormatException e) {
            e.printStackTrace();
            Alert alert = new Alert(
                    Alert.AlertType.ERROR,
                     "input must be Integer",
                    ButtonType.OK
            );
            alert.showAndWait();
            return Integer.MIN_VALUE;
        }
    }

    public void disableAll() {
        readbt.setDisable(true);
        nodeIdField.setDisable(true);
        netcb.setDisable(true);
        levelcb.setDisable(true);
        activatorPane.setDisable(true);
        opcb.setDisable(true);
        losscb.setDisable(true);
        opPane.setDisable(true);
        lossPane.setDisable(true);
        lastPane.setDisable(true);
    }

    @FXML
    public void generateCode() {
        // 保存路径/output/log1/node1/....

        LocalTime time = LocalTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        String infoText = infoArea.getText();
        String currentTime = time.format(formatter);
        infoText += currentTime + " 正在生成模型......\n";
        infoArea.setText(infoText);

        String logOutputString = outputString + "/" + logName;
        File logOutputDir = new File(logOutputString);
        if (!logOutputDir.exists()) {
            logOutputDir.mkdir();
        }
        String nodeOutputString = logOutputString + "/node" + wrapper.getNodeId();
        File nodeOutputDir = new File(nodeOutputString);
        if (!nodeOutputDir.exists()) {
            nodeOutputDir.mkdir();
        }
        String fileName = nodeOutputString + "/sample.py";
        mkFile(fileName);
        String codeText = wrapper.toText();
//        System.out.println(codeText);
        try (FileWriter writer = new FileWriter(new File(fileName))) {
            writer.write(codeText);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Process process;
        try {
            String modelAbsolutePath = new File(fileName).getAbsolutePath();
            String cmdstr = "python process.py " + modelAbsolutePath;
            File dir = new File(currentPath + "\\model");
//            System.out.println(cmdstr);
            process = Runtime.getRuntime().exec(cmdstr, null, dir);
            process.waitFor();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        infoText = infoArea.getText();
        currentTime = time.format(formatter);
        infoText += currentTime + " 成功生成模型\n";
        infoArea.setText(infoText);

        trainbt.setDisable(false);
    }

    @FXML
    public void trainModel() {

        LocalTime time = LocalTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        String infoText = infoArea.getText();
        String currentTime = time.format(formatter);
        infoText += currentTime + " 正在训练模型......\n";
        infoArea.setText(infoText);

        Process process;
        try {
            String logDir = outputString + "/" + logName;
            File logDirFile = new File(logDir);
            String logAbsolutePath = logDirFile.getAbsolutePath();
            File dir = new File(currentPath + "\\model");
            String cmdstr = "python Complete_Training.py " + logAbsolutePath + " " + wrapper.getNodeId();
            process = Runtime.getRuntime().exec(cmdstr, null, dir);
            process.waitFor();

            infoText = infoArea.getText();
            currentTime = time.format(formatter);
            infoText += currentTime + " 已训练模型\n";
            infoArea.setText(infoText);
        } catch (Throwable e) {
            e.printStackTrace();
        }
        testbt.setDisable(false);
    }

    @FXML
    public void testModel() {
        Process process;
        LocalTime time = LocalTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        String infoText = infoArea.getText();
        String currentTime = time.format(formatter);
        infoText += currentTime + " 正在测试模型......\n";
        infoArea.setText(infoText);

        try {
            String logDir = outputString + "/" + logName;
            File logDirFile = new File(logDir);
            String logAbsolutePath = logDirFile.getAbsolutePath();
            File dir = new File(currentPath + "\\model");
            String cmdstr = "python Complete_Testing.py " + logAbsolutePath + "\\node" + wrapper.getNodeId();
            process = Runtime.getRuntime().exec(cmdstr, null, dir);
            process.waitFor();

            infoText = infoArea.getText();
            currentTime = time.format(formatter);
            infoText += currentTime + " 已测试模型，点击测试报告tab查看\n";
            infoArea.setText(infoText);
        } catch (Throwable e) {
            e.printStackTrace();
        }

    }

    // 从dot文件获得节点id和它对应的节点
    private Map<Integer, String> getIdMap(String filePath) {
//        File dot = new File("C:\\H\\Java_codes\\output\\solution12_small\\diagram.dot");
        File dot = new File(filePath);
        StringBuilder builder = new StringBuilder();
        Map<Integer, String> map = new HashMap<>();
        try {
            FileInputStream in = new FileInputStream(dot);
            int n = 0;
            // n == -1 代表读到文件末尾
            while (n != -1) {
                n = in.read();
                char ch = (char) n;
                builder.append(ch);
            }
        } catch (Throwable e) {
            e.printStackTrace();
        }
        String[] strs = builder.toString().split("\"");
        for (int i = 0; i < strs.length; i++) {
            if (i % 2 == 1) {
                if (!strs[i].equals(" ----> ")) {
                    String[] parts = strs[i].split("\n");
                    String[] tmp = parts[0].split(" ");
                    int node_id = Integer.parseInt(tmp[2].trim());
                    map.put(node_id, parts[1]);
                }
            }
        }
//        for (Integer key : map.keySet()) {
//            System.out.println(key + " " + map.get(key));
//        }
        return map;
    }

    private void mkFile(String fileName) {
        //TODO
        File file = new File(fileName);
        try {
            file.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
            Alert alert = new Alert(
                    Alert.AlertType.ERROR,
                    "file already exists!",
                    ButtonType.OK,
                    ButtonType.CANCEL
            );
            alert.showAndWait();
        }
    }

    public void init(Stage primaryStage) {
        this.wrapper = new Wrapper();

        opMap.put("SGD", 5);
        opMap.put("ASGD", 5);
        opMap.put("Rprop", 3);
        opMap.put("Adagrad", 3);
        opMap.put("Adadelta", 4);
        opMap.put("RMSprop", 6);
        opMap.put("Adam", 5);
        opMap.put("Adamax", 4);

        lossMap.put("BCELoss", 4);
        lossMap.put("BCEWithLogitsLoss", 5);
        lossMap.put("NLLLoss", 4);
        lossMap.put("CrossEntropyLoss", 2);
        lossMap.put("L1Loss", 3);
        lossMap.put("MSELoss", 3);
        lossMap.put("SmoothL1Loss", 4);

        activatorPane.setVisible(false);
        opPane.setVisible(false);
        lossPane.setVisible(false);
        lastPane.setVisible(false);
        generate.setDisable(true);
        trainbt.setDisable(true);
        testbt.setDisable(true);
        this.stage = primaryStage;

        File outputDir = new File(outputString);
        if (!outputDir.exists()) {
            outputDir.mkdir();
        }

        outBox.setHgrow(ap, Priority.ALWAYS);
    }

    @FXML
    public void showImage() {
        if (pngPath != null && new File(pngPath).exists()) {
            try {
                FileInputStream input = new FileInputStream(pngPath);
                Image image = new Image(input);
                imageView.setImage(image);
                // 图片宽度随窗口大小变化
                imageView.fitWidthProperty().bind(stage.widthProperty());
            } catch (Throwable e) {
                e.printStackTrace();
            }
        }
    }

    @FXML
    public void showReport() {
        String nodeDir = logAbsoluteDir + "node" + wrapper.getNodeId();
        String reportDir = nodeDir + "\\report.txt";
        File reportFile = new File(reportDir);
        if (reportFile.exists()) {
            try {
                InputStreamReader isr = new InputStreamReader(new FileInputStream(reportFile), "UTF-8");
                BufferedReader br = new BufferedReader(isr);
                String line;
                StringBuilder reportText = new StringBuilder("");
                while ((line = br.readLine()) != null) {
                    reportText.append(line);
                    reportText.append("\n");
                }
                reportArea.setText(reportText.toString());
            } catch (Throwable e) {
                e.printStackTrace();
            }
        }
    }

}
