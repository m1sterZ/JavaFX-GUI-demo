package demo;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextField;
import javafx.scene.layout.AnchorPane;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import java.io.File;

public class Controller {
    private Stage stage;
    private FileChooser fileChooser;
    private Wrapper wrapper;
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
    public void selectOp() {
        String op = opcb.getValue();
        opText1.setText("lr:");
        switch (op) {
            case "1":
                opText2.setText("param:");
                opText3.setText("param:");
                opText4.setText("param:");
                opText5.setText("param:");
                opText6.setText("param:");
                break;
            case "2":
                opText5.setVisible(false);
                opParam5.setVisible(false);
                opText6.setVisible(false);
                opParam6.setVisible(false);
                break;
        }
        wrapper.setOptimizer(op);
    }

    //SGD
    //ASGD
    //Adam
    //Adamax
    //Adadelta
    //Adagrad
    //Rprop
    //RMSprop

    @FXML
    public void selectLoss() {
        String loss = losscb.getValue();
        switch (loss) {
            case "BCELoss":
                lossPane.setVisible(true);
                lossText1.setText("weight:");
                lossText1.setVisible(true);
                lossParam1.setVisible(true);
                lossText2.setText("size_average:");
                lossText2.setVisible(true);
                lossParam2.setVisible(true);
                lossText3.setText("reduce:");
                lossText3.setVisible(true);
                lossParam3.setVisible(true);
                lossText4.setText("reduction:");
                lossText4.setVisible(true);
                lossParam4.setVisible(true);
                lossText5.setVisible(false);
                lossParam5.setVisible(false);
                break;
            case "BCEWithLogitsLoss":
                lossPane.setVisible(true);
                lossText1.setText("weight:");
                lossText1.setVisible(true);
                lossParam1.setVisible(true);
                lossText2.setText("size_average:");
                lossText2.setVisible(true);
                lossParam2.setVisible(true);
                lossText3.setText("reduce:");
                lossText3.setVisible(true);
                lossParam3.setVisible(true);
                lossText4.setText("reduction:");
                lossText4.setVisible(true);
                lossParam4.setVisible(true);
                lossText5.setText("pos_weight:");
                lossText5.setVisible(true);
                lossParam5.setVisible(true);
                break;
            case "NLLLoss":
                lossPane.setVisible(true);
                lossText1.setText("weight:");
                lossText1.setVisible(true);
                lossParam1.setVisible(true);
                lossText2.setText("size_average:");
                lossText2.setVisible(true);
                lossParam2.setVisible(true);
                lossText3.setText("ignore_index:");
                lossText3.setVisible(true);
                lossParam3.setVisible(true);
                lossText4.setText("reduce:");
                lossText4.setVisible(true);
                lossParam4.setVisible(true);
                lossText5.setText("reduction:");
                lossText5.setVisible(true);
                lossParam5.setVisible(true);
                break;
            case "SmoothL1Loss":
            case "L1Loss":
                lossPane.setVisible(true);
                lossText1.setText("size_average:");
                lossText1.setVisible(true);
                lossParam1.setVisible(true);
                lossText2.setText("reduce:");
                lossText2.setVisible(true);
                lossParam2.setVisible(true);
                lossText3.setText("reduction:");
                lossText3.setVisible(true);
                lossParam3.setVisible(true);
                lossText4.setVisible(false);
                lossParam4.setVisible(false);
                lossText5.setVisible(false);
                lossParam5.setVisible(false);
                break;
            case "CrossEntropyLoss":
            case "MSELoss":
                lossPane.setVisible(false);
                break;
        }
        wrapper.setLossFunc(loss);
        levelPane.setVisible(true);
    }
//torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
//torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
//torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
//torch.nn.CrossEntropyLoss()
//torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
//torch.nn.MSELoss()
//torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean')


    @FXML
    public void selectActivator() {
        String activator = activatorcb.getValue();
        wrapper.setActivator(activator);
    }

    @FXML
    public void saveAll() {
        // TODO: save all settings from textfields into wrapper

        generate.setDisable(false);
    }

    @FXML
    public void generateCode() {
        // TODO: check if model file exists
        //  alert and generate model
    }

    public void init(Stage primaryStage) {
        this.wrapper = new Wrapper();
        opPane.setVisible(false);
        lossPane.setVisible(false);
        levelPane.setVisible(false);
        generate.setDisable(true);
        this.stage = primaryStage;
    }

}
