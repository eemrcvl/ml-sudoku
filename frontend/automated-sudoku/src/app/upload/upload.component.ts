import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormBuilder, FormGroup } from '@angular/forms';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.css']
})
export class UploadComponent implements OnInit {
  uploadForm: FormGroup;
  API_URL = "http://localhost:5000/";
  rows: number[] = [];
  show: boolean = false;

  constructor(private httpClient: HttpClient, private formBuilder: FormBuilder) {
   }

  ngOnInit(): void {
    this.uploadForm = this.formBuilder.group({
      profile: ['']
    });
  }

  onFileSelect(event) {
    if(event.target.files.length > 0){
      const file = event.target.files[0];
      this.uploadForm.get('profile').setValue(file);
    }
    /*const formData = new FormData();

    formData.append('file', this.uploadForm.get('profile').value);

    this.httpClient.post<any>(this.API_URL, formData).subscribe(
      (res: number[][]) => this.board = res,
      (err) => console.log(err)
    );
      console.log(this.board[0]);*/
  }

  onSubmit() {
    this.show = !this.show;
    const formData = new FormData();

    formData.append('file', this.uploadForm.get('profile').value);

    this.httpClient.post<any>(this.API_URL, formData).subscribe(
      (res) => this.showBoard(res),
      (err) => console.log(err)
    );
  }

  showBoard(result){
    if (typeof result === 'string') {
      alert("Not a Sudoku puzzle, try again");
      window.location.href="/solve";
    }
    else {
    for(var i=0; i < 9; i++){
      for(var j=0; j < 9; j++){
        this.rows.push(result[i][j]);
      }
    }
  }

  }


}
