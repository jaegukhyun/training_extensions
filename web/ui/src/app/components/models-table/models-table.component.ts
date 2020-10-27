/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Component, EventEmitter, OnDestroy, OnInit, Output, ViewChild} from '@angular/core';
import {MatSort} from '@angular/material/sort';
import {MatTableDataSource} from '@angular/material/table';
import {IMetric, IModel} from '@idlp/root/models';
import {
  IBuild,
  IModelsMetricsTableColumn,
  IModelsMetricsTableData
} from '@idlp/routed/problem-info/problem-info.models';
import {ProblemInfoState} from '@idlp/routed/problem-info/problem-info.state';
import {Utils} from '@idlp/utils/utils';
import {Select} from '@ngxs/store';
import {Observable, Subject} from 'rxjs';


@Component({
  selector: 'idlp-models-table',
  templateUrl: './models-table.component.html',
  styleUrls: ['./models-table.component.scss']
})
export class IdlpModelsTableComponent implements OnDestroy, OnInit {
  @ViewChild(MatSort, {static: true})
  sort: MatSort;

  @Select(ProblemInfoState.activeBuild) activeBuild$: Observable<IBuild>;
  @Select(ProblemInfoState.modelsMetricsTableData) modelsMetricsTableData$: Observable<IModelsMetricsTableData[]>;
  @Select(ProblemInfoState.modelsMetricsTableColumns) modelsMetricsTableColumns$: Observable<IModelsMetricsTableColumn[]>;
  @Select(ProblemInfoState.models) models$: Observable<IModel[]>;

  problemId: string;

  @Output()
  onMenuItemClick: EventEmitter<any> = new EventEmitter<any>();

  metrics: IMetric[];
  selectedModel: IModel;

  displayColumns: string[];
  columns: IModelsMetricsTableColumn[];

  data: IModelsMetricsTableData[];
  dataSource: MatTableDataSource<{ [metric: string]: string }>;

  private models: IModel[] = [];

  private destroy$: Subject<any> = new Subject();

  private activeBuild: IBuild;

  isBuildValidForEvaluate(modelId: string): boolean {
    if (this.activeBuild.status === 'default') {
      return false;
    }
    const model = this.models.find((m: IModel) => m.id === modelId);
    return !(model.metrics && this.activeBuild.id in model.metrics);
  }

  ngOnInit(): void {
    this.activeBuild$.subscribe((activeBuild: IBuild) => this.activeBuild = activeBuild);
    this.modelsMetricsTableColumns$.subscribe((columns: IModelsMetricsTableColumn[]) => {
      this.columns = columns;
      this.displayColumns = this.columns.map(column => column.key);
    });

    this.modelsMetricsTableData$.subscribe((data) => {
      this.data = data;
      this.dataSource = new MatTableDataSource(data);
      this.dataSource.sort = this.sort;
      this.dataSource.sortData = (d: { [metric: string]: any }[], sort: MatSort): { [metric: string]: any }[] => {
        d.sort((a, b) => {
          if (!a[sort.active]?.value || !b[sort.active]?.value) {
            return 0;
          }
          if (a[sort.active].value < b[sort.active].value) {
            return sort.direction === 'asc' ? 1 : -1;
          }
          if (a[sort.active].value > b[sort.active].value) {
            return sort.direction === 'asc' ? -1 : 1;

          }
          return 0;
        });
        return d;
      };
    });
    this.models$.subscribe((models) => this.models = models);
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  isEmpty(value: any): boolean {
    return Utils.isEmpty(value);
  }

  isHiddenOnChart(model: IModel): boolean {
    const hoc: string[] = JSON.parse(localStorage.getItem('hoc'));
    return hoc?.includes(model.id);
  }

  getCellStyle(opacity: number): string {
    if (opacity !== undefined) {
      return `background-color: rgba(38, 134, 206, ${opacity})`;
    }
  }

  isContrastText(opacity: number): boolean {
    if (opacity === undefined) {
      return false;
    }
    return opacity > 0.4;
  }

  menuItemClick(action: string): void {
    this.onMenuItemClick.emit({action, model: this.selectedModel});
  }

  rowClick(item: IModel): void {
    this.selectedModel = this.models.filter((model: IModel) => model.id === item.id)[0];
  }
}
