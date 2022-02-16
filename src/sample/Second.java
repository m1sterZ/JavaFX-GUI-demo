package sample;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class Second extends Application {
    Stage secondStage = new Stage();

    @Override
    public void start(Stage primaryStage) throws Exception{
        Parent root = FXMLLoader.load(getClass().getResource("second.fxml"));
        primaryStage.setTitle("JavaFX demo");
        primaryStage.setScene(new Scene(root, 600, 400));
        primaryStage.show();
    }

    public void showWindow() throws Exception{
        start(secondStage);
    }

    public static void main(String[] args) {
        launch(args);
    }
}
