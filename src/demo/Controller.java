package demo;

import com.sun.xml.internal.ws.util.StringUtils;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.layout.AnchorPane;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import java.io.File;
import java.text.NumberFormat;
import java.util.*;

public class Controller {
    private Stage stage;
    public FileChooser fileChooser = new FileChooser();
    private Wrapper wrapper;
    private Map<String, Integer> opMap = new HashMap<>();
    private Map<String, Integer> lossMap = new HashMap<>();
    @FXML
    public AnchorPane ap;
    public Button readbt;
    public ComboBox<String> netcb;
    public ComboBox<String> activatorcb;
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
    public AnchorPane levelPane;
    public TextField levelField;
    public TextField epochField;
    public TextField inDiField;
    public TextField outDiField;
    public Button saveSetting;
    public Button generate;


    @FXML
    public void readFile() {
        fileChooser.setTitle("open a file");
        File file = fileChooser.showOpenDialog(stage);
    }

    @FXML
    public void selectNet() {
        String netName = netcb.getValue();
        wrapper.setNetName(netName);
    }

    @FXML
    public void selectActivator() {
        String activator = activatorcb.getValue();
        wrapper.setActivator(activator);
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
        levelPane.setVisible(true);
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
                "Save all settings?",
                ButtonType.YES,
                ButtonType.NO,
                ButtonType.CANCEL
        );
        saveAlert.setTitle("Confirm");
        saveAlert.showAndWait();
        if (saveAlert.getResult() == ButtonType.YES) {
            //TODO: save all settings from textfields into wrapper
            disableAll();
            saveOp();
            saveLoss();
            saveLevels();
            generate.setDisable(false);
        }
    }

    public void saveOp() {
        String op = wrapper.getOptimizer();
        int num = opMap.get(op);
        String[] opParams = new String[num];
        switch (op) {
            case "SGD":
            case "ASGD":
            case "Rprop":
            case "Adagrad":
            case "Adadelta":
            case "Adam":
            case "Adamax":
            case "RMSprop":
        }
        wrapper.setOpParams(opParams);
    }

    public void saveLoss() {
        String loss = wrapper.getLossFunc();
        int num = lossMap.get(loss);
        String[] lossParams = new String[num];
        switch (loss) {
            case "BCELoss":
            case "NLLLoss":
            case "BCEWithLogitsLoss":
            case "L1Loss":
            case "MSELoss":
            case "CrossEntropyLoss":
            case "SmoothL1Loss":
        }
        wrapper.setLossParams(lossParams);
    }

    public void saveLevels() {
        String levelstr = levelField.getText();
        String epochstr = epochField.getText();
        int l = isInt(levelstr,"level");
        int e = isInt(epochstr, "epoch");
        if (l > 0) wrapper.setLevel(l);
        if (e > 0) wrapper.setEpoch(e);
        String[] inDimensions = inDiField.getText().split(",");
        String[] outDimensions = outDiField.getText().split(",");
        List<Integer> list = new ArrayList<>();
        for (String s : inDimensions) {
            int res = isInt(s, "in_dimension");
            //TODO handle invalid input
            if (res > Integer.MIN_VALUE) list.add(res);
        }
        int[] tmpArray = new int[l];
        //
        list.clear();
        for (String s : outDimensions) {
            int res = isInt(s, "out_dimensions");
            if (res > Integer.MIN_VALUE) list.add(res);
        }
        //
    }

    private int isInt(String str, String paraName) {
        try {
            int res = Integer.parseInt(str);
            return res;
        } catch (NumberFormatException e) {
            e.printStackTrace();
            Alert alert = new Alert(
                    Alert.AlertType.ERROR,
                     paraName + " must be Integer",
                    ButtonType.OK
            );
            alert.showAndWait();
            return Integer.MIN_VALUE;
        }
    }

    public void disableAll() {
        readbt.setDisable(true);
        netcb.setDisable(true);
        activatorcb.setDisable(true);
        opcb.setDisable(true);
        losscb.setDisable(true);
        opPane.setDisable(true);
        lossPane.setDisable(true);
        levelPane.setDisable(true);
    }

    @FXML
    public void generateCode() {
        // TODO: check if model file exists
        //  alert and generate model
        Alert genAlert = new Alert(
                Alert.AlertType.CONFIRMATION,
                "Continue to generate model?",
                ButtonType.YES,
                ButtonType.NO,
                ButtonType.CANCEL
        );
        genAlert.setTitle("Confirm");
        genAlert.showAndWait();
        if (genAlert.getResult() == ButtonType.YES) {
//            Platform.exit();
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


        opPane.setVisible(false);
        lossPane.setVisible(false);
        levelPane.setVisible(false);
        generate.setDisable(true);
        this.stage = primaryStage;
    }

}
