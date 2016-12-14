function [ output_args ] = getidentityid( tmp )
% Identity Label , return the id of indentity
switch tmp
  case 'KA'
      output_args = 1;
  case 'KL'
      output_args = 2;
  case 'KM'
      output_args = 3;
  case 'KR'
      output_args = 4;
  case 'MK'
      output_args = 5;
  case 'NA'
      output_args = 6;
  case 'NM'
      output_args = 7;
  case 'TM'
      output_args = 8;
  case 'UY'
      output_args = 9;
  case 'YM'
      output_args = 10;
end

end

