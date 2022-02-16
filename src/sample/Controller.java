package sample;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;

import java.net.URL;
import java.util.ResourceBundle;

public class Controller implements Initializable {

    public AnchorPane root;
    private Stage stage;

    public Controller() { }

    private Stage getStage() {
        stage = (Stage) root.getScene().getWindow();
        return stage;
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {

    }

    @FXML
    public void changeWindow() throws Exception {
        Stage current = getStage();
        current.close();
        Second second = new Second();
        second.showWindow();
    }
}
