
import java.util.ArrayList;
import java.util.List;

class Solution {

    public int[] getRow(int[][] reservedSeats, int row) {
        List<Integer> seatRowNums = new ArrayList<>();
        for (int[] seat : reservedSeats) {
            if (seat[0] == row) {
                seatRowNums.add(seat[1]);
            }
        }

        int[] result = new int[seatRowNums.size()];
        for (int i = 0; i < seatRowNums.size(); i++) {
            result[i] = seatRowNums.get(i);
        }

        return result;
    }

    public boolean arrayContains(int[] array, int elem) {
        for (int i = 0; i < array.length; i++) {
            if (array[i] == elem) {
                return true;
            }
        }
        return false;
    }

    public int checkDouble(int[][] reservedSeats, int row) {
        int[] myRow = this.getRow(reservedSeats, row);
        int count = 2;
        if (this.arrayContains(myRow, 2)
                || this.arrayContains(myRow, 3)
                || this.arrayContains(myRow, 4)
                || this.arrayContains(myRow, 5)) {
            count -= 1;
        }
        if (this.arrayContains(myRow, 6)
                || this.arrayContains(myRow, 7)
                || this.arrayContains(myRow, 8)
                || this.arrayContains(myRow, 9)) {
            count -= 1;
        }
        return count;
    }

    public int checkMiddle(int[][] reservedSeats, int row) {
        int[] myRow = this.getRow(reservedSeats, row);
        if (this.arrayContains(myRow, 4)
                || this.arrayContains(myRow, 5)
                || this.arrayContains(myRow, 6)
                || this.arrayContains(myRow, 7)) {
            return 0;
        }
        return 1;

    }

    public int getRowCount(int[][] reservedSeats, int row) {
        int doubleCount = this.checkDouble(reservedSeats, row);
        if (doubleCount > 0) {
            int middleCount = this.checkMiddle(reservedSeats, row);
            return middleCount;
        }
        return doubleCount;
    }

    public int maxNumberOfFamilies(int n, int[][] reservedSeats) {

        int count = 0;
        for (int row = 1; row <= n; row++) {
            count += this.getRowCount(reservedSeats, row);
        }

        return count;

    }

    public static void Main(String[] args) {
        Solution s = new Solution;
        System.out.println(s.getRow([[1,2],[1,3],[1,8],[2,6],[3,1],[3,10]], 1));
    }

}
