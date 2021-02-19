## <center> SUMMARY
This dataset was constructed by YOOCHOOSE to support participants in the RecSys Challenge 2015 (http://recsys.yoochoose.net).   
The YOOCHOOSE dataset contain a collection of sessions from a retailer, where each session
is encapsulating the click events that the user performed in the session.
For some of the sessions, there are also buy events.
    
### CLICKS DATASET
The file yoochoose-clicks.dat comprising the clicks of the users over the items.
Each record/line in the file has the following fields/format: Session ID, Timestamp, Item ID, Category
* Session ID – the id of the session. In one session there are one or many clicks. Could be represented as an integer number.
* Timestamp – the time when the click occurred. Format of YYYY-MM-DDThh:mm:ss.SSSZ
* Item ID – the unique identifier of the item that has been clicked. Could be represented as an integer number.
* Category – the context of the click. The value "S" indicates a special offer, "0" indicates  a missing value, a number between 1 to 12 indicates a real category identifier, any other number indicates a brand. E.g. if an item has been clicked in the context of a promotion or special offer then the value will be "S", if the context was a brand i.e BOSCH, then the value will be an 8-10 digits number. If the item has been clicked under regular category, i.e. sport, then the value will be a number between 1 to 12. 
 
### BUYS DATSET
The file yoochoose-buys.dat comprising the buy events of the users over the items.
Each record has the following fields:
* Session ID - the id of the session. In one session there are one or many buying events. Could be represented as an integer number.
* Timestamp - the time when the buy occurred. Format of YYYY-MM-DDThh:mm:ss.SSSZ
* Item ID – the unique identifier of item that has been bought. Could be represented as an integer number.
* Price – the price of the item. Could be represented as an integer number.
* Quantity – the quantity in this buying.  Could be represented as an integer number.

